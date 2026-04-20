from typing import Any

import jax
import jax.numpy as jnp
import pytest
from benchmark_utils import AUTOTUNE_CACHE_DIR, bench, peak_hbm_mb
from flax import nnx
from jax.experimental import mesh_utils
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float
from PIL import Image
from transformers import AutoConfig, ViTForImageClassification, ViTImageProcessor

import jimm
from jimm import VisionTransformer

HF_MODEL_NAME = "google/vit-base-patch16-224"

devices = mesh_utils.create_device_mesh((jax.device_count(), 1))
mesh = Mesh(devices, ("data", "fsdp"))
jax.set_mesh(mesh)


@jax.jit
def create_model() -> VisionTransformer:
    """Create and shard ViT model.

    Returns:
        VisionTransformer: Sharded model.
    """
    model = VisionTransformer.from_pretrained(HF_MODEL_NAME, rngs=nnx.Rngs(0))
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    nnx.update(model, jax.lax.with_sharding_constraint(state, pspecs))
    return model


def _forward(model: VisionTransformer, image: Float[Array, "batch height width channels"]) -> Float[Array, "batch num_classes"]:
    return model(image)


forward = nnx.jit(_forward)


def test_vision_transformer_inference() -> None:
    """Run ViT inference and compare to HF reference.

    Returns:
        None
    """
    global mesh
    with mesh:
        model = create_model()
    image = Image.open("images/test_image.jpg")
    processor = ViTImageProcessor.from_pretrained(HF_MODEL_NAME)
    inputs = processor(images=image, return_tensors="pt")

    pytorch_model = ViTForImageClassification.from_pretrained(HF_MODEL_NAME)
    pytorch_model.eval()
    outputs = pytorch_model(**inputs)
    logits_ref = outputs.logits.detach().cpu().numpy()

    model.eval()
    x_eval: Float[Array, "batch height width channels"] = jnp.transpose(inputs["pixel_values"].detach().cpu().numpy(), axes=(0, 2, 3, 1))
    logits_flax = forward(model, x_eval)
    print(f"Max absolute difference: {jnp.abs(logits_flax - logits_ref).max()}")
    assert jnp.allclose(logits_flax, logits_ref, atol=0.05)


def test_vision_transformer_from_config() -> None:
    """Test VisionTransformer.from_config creates model with correct architecture.

    Returns:
        None
    """
    config = AutoConfig.from_pretrained(HF_MODEL_NAME).to_dict()
    model = VisionTransformer.from_config(config, rngs=nnx.Rngs(0))
    x = jnp.ones((1, config["image_size"], config["image_size"], 3))
    output = model(x)
    num_classes = len(config["id2label"]) if "id2label" in config else config.get("num_labels", 1000)
    assert output.shape == (1, num_classes)


test_vision_transformer_inference()
test_vision_transformer_from_config()


def test_vit_explicit_sharding() -> None:
    """Test ViT forward pass in JAX explicit sharding mode."""

    n_devices = jax.device_count()
    explicit_devices = mesh_utils.create_device_mesh((n_devices, 1))
    explicit_mesh = Mesh(explicit_devices, ("data", "fsdp"), axis_types=(AxisType.Explicit, AxisType.Explicit))

    config = AutoConfig.from_pretrained(HF_MODEL_NAME).to_dict()
    traced_specs: dict[str, P] = {}

    with explicit_mesh:
        model = VisionTransformer.from_config(config, rngs=nnx.Rngs(0))
        model.eval()

        @nnx.jit
        def forward(model: VisionTransformer, image: Float[Array, "batch h w c"]) -> Float[Array, "batch num_classes"]:
            traced_specs["image"] = jax.typeof(image).sharding.spec
            traced_specs["vision_pos_embed"] = jax.typeof(model.encoder.position_embeddings[...]).sharding.spec
            logits = model(image)
            traced_specs["output"] = jax.typeof(logits).sharding.spec
            return logits

        image = jax.device_put(
            jnp.ones((n_devices, config["image_size"], config["image_size"], 3)),
            NamedSharding(explicit_mesh, P("data", None, None, None)),
        )
        out = jax.block_until_ready(forward(model, image))

    assert out.shape == (n_devices, config.get("num_labels", 1000))
    assert traced_specs["image"][0] == "data", f"image batch dim not sharded on 'data': {traced_specs['image']}"
    assert traced_specs["vision_pos_embed"] == P(None, None, "fsdp"), f"unexpected vision positional embedding sharding: {traced_specs['vision_pos_embed']}"
    assert traced_specs["output"] == P("data", None), f"unexpected logits sharding: {traced_specs['output']}"


@pytest.mark.parametrize("batch_size_per_device", [8, 16, 32, 64])
def test_vit_tokamax_attention(batch_size_per_device: int, hf_model_name: str = HF_MODEL_NAME) -> None:
    """Test VisionTransformer with tokamax attention backends.

    ViT-base-patch16-224 has 197 tokens (196 patches + CLS), padded to 256 for
    mosaic_tpu. Parametrized over batch sizes.

    Args:
        batch_size_per_device (int): Samples per device.
        hf_model_name (str): HuggingFace model name. Defaults to HF_MODEL_NAME.

    Returns:
        None
    """
    from jax.sharding import NamedSharding

    n_devices = jax.device_count()
    total_batch = batch_size_per_device * n_devices

    config = AutoConfig.from_pretrained(hf_model_name).to_dict()
    img_size = config["image_size"]
    patch_size = config["patch_size"]
    seq_len = (img_size // patch_size) ** 2 + 1

    image_sharding = NamedSharding(mesh, P("data", None, None, None))
    image = jax.device_put(
        jnp.ones((total_batch, img_size, img_size, 3)),
        image_sharding,
    )

    print(f"\nModel: {hf_model_name}  (seq={seq_len}, padded_to={-(-seq_len // 128) * 128})")
    mosaic = jimm.make_tokamax_attention(["mosaic_tpu", "xla_chunked"])
    models = {
        "standard": VisionTransformer.from_config(config, rngs=nnx.Rngs(0)),
        "mosaic": VisionTransformer.from_config(config, rngs=nnx.Rngs(0), attention_fn=mosaic),
        "chunked": VisionTransformer.from_config(config, rngs=nnx.Rngs(0), attention_fn=jimm.make_tokamax_attention("xla_chunked")),
        "xla": VisionTransformer.from_config(config, rngs=nnx.Rngs(0), attention_fn=jimm.make_tokamax_attention("xla")),
    }
    for m in models.values():
        m.eval()

    @nnx.jit
    def forward(model: VisionTransformer, image: Float[Array, "batch height width channels"]) -> Float[Array, "batch num_classes"]:
        return model(image)

    tuned = {k: jimm.autotuned_fn(forward, models[k], image, cache_dir=AUTOTUNE_CACHE_DIR) for k in models}
    outs = {k: jax.block_until_ready(tuned[k](models[k], image)) for k in models}
    ms = {k: bench(tuned[k], models[k], image) for k in models}
    peaks = {k: peak_hbm_mb(forward, models[k], image) for k in models}

    print(f"batch_size={total_batch}  ({batch_size_per_device} per device × {n_devices} devices)")
    ref = ms["standard"]
    for k, label in [
        ("standard", "Standard:                     "),
        ("mosaic", "Tokamax (mosaic_tpu→chunked): "),
        ("chunked", "Tokamax (xla_chunked):        "),
        ("xla", "Tokamax (xla):                "),
    ]:
        speed = f"  ({ref / ms[k]:.2f}x speed)" if k != "standard" else ""
        print(f"{label}{ms[k]:.2f} ms/fwd  peak HBM: {peaks[k]:.1f} MB{speed}")
    for k in ("mosaic", "chunked", "xla"):
        print(f"max diff {k:<10} vs standard: {jnp.abs(outs[k] - outs['standard']).max():.2e}")

    for k in ("mosaic", "chunked", "xla"):
        assert jnp.allclose(outs["standard"], outs[k], atol=2e-2), f"{k} outputs differ: {jnp.abs(outs['standard'] - outs[k]).max()}"


@pytest.mark.parametrize("batch_size_per_device", [4, 8, 16])
def test_vit_long_context_attention(batch_size_per_device: int) -> None:
    """Benchmark splash attention on ViT with image_size=512, patch_size=16.

    Gives 1024+1=1025 tokens (padded to 1152 = 9×128) — enough for mosaic_tpu
    to overcome its overhead. Tests forward and backward pass.

    Args:
        batch_size_per_device (int): Samples per device.

    Returns:
        None
    """
    from jax.sharding import NamedSharding

    n_devices = jax.device_count()
    total_batch = batch_size_per_device * n_devices

    config = {
        "image_size": 512,
        "patch_size": 16,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "num_labels": 1000,
    }
    img_size = config["image_size"]
    seq_len = (img_size // config["patch_size"]) ** 2 + 1

    image_sharding = NamedSharding(mesh, P("data", None, None, None))
    image = jax.device_put(
        jnp.ones((total_batch, img_size, img_size, 3)),
        image_sharding,
    )

    print(f"\nViT long-context (seq={seq_len}, padded_to={-(-seq_len // 128) * 128})")
    mosaic_fwd = jimm.make_tokamax_attention(["mosaic_tpu", "xla_chunked"])
    mosaic_bwd = jimm.make_tokamax_attention(["mosaic_tpu", "xla"])
    models_fwd = {
        "standard": VisionTransformer.from_config(config, rngs=nnx.Rngs(0)),
        "mosaic": VisionTransformer.from_config(config, rngs=nnx.Rngs(0), attention_fn=mosaic_fwd),
    }
    models_bwd = {
        "standard": VisionTransformer.from_config(config, rngs=nnx.Rngs(0)),
        "mosaic": VisionTransformer.from_config(config, rngs=nnx.Rngs(0), attention_fn=mosaic_bwd),
    }
    for m in {**models_fwd, **models_bwd}.values():
        m.eval()

    @nnx.jit
    def forward(model: VisionTransformer, image: Float[Array, "batch height width channels"]) -> Float[Array, "batch num_classes"]:
        return model(image)

    @nnx.jit
    def backward(model: VisionTransformer, image: Float[Array, "batch height width channels"]) -> Any:
        return nnx.grad(lambda m: jnp.mean(m(image)))(model)

    tuned_fwd = {k: jimm.autotuned_fn(forward, models_fwd[k], image, cache_dir=AUTOTUNE_CACHE_DIR) for k in models_fwd}
    outs = {k: jax.block_until_ready(tuned_fwd[k](models_fwd[k], image)) for k in models_fwd}
    fwd_ms = {k: bench(tuned_fwd[k], models_fwd[k], image) for k in models_fwd}
    bwd_ms = {k: bench(backward, models_bwd[k], image) for k in models_bwd}
    peaks = {k: peak_hbm_mb(forward, models_fwd[k], image) for k in models_fwd}

    print(f"batch_size={total_batch}  ({batch_size_per_device} per device × {n_devices} devices)")
    ref_fwd, ref_bwd = fwd_ms["standard"], bwd_ms["standard"]
    for k, label in [
        ("standard", "Standard:                     "),
        ("mosaic", "Tokamax (mosaic_tpu→chunked): "),
    ]:
        fwd_speed = f"  ({ref_fwd / fwd_ms[k]:.2f}x)" if k != "standard" else ""
        bwd_speed = f"  ({ref_bwd / bwd_ms[k]:.2f}x)" if k != "standard" else ""
        print(f"{label}fwd {fwd_ms[k]:.2f} ms{fwd_speed}  bwd {bwd_ms[k]:.2f} ms{bwd_speed}  peak HBM: {peaks[k]:.1f} MB")
    print(f"max diff mosaic vs standard: {jnp.abs(outs['mosaic'] - outs['standard']).max():.2e}")

    assert jnp.allclose(outs["standard"], outs["mosaic"], atol=5e-2), f"mosaic outputs differ: {jnp.abs(outs['standard'] - outs['mosaic']).max()}"
