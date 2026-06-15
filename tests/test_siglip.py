import jax
import jax.numpy as jnp
import pytest
from benchmark_utils import AUTOTUNE_CACHE_DIR, bench, peak_hbm_mb
from flax import nnx
from jax.experimental import mesh_utils
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoProcessor, SiglipTextModel, SiglipVisionModel

import jimm
from jimm import SigLIP, SigLIPTextModel, SigLIPVisionModel
from jimm.common.sharding import NoSharding
from jimm.models.siglip.sharding import SigLIPSharding

HF_MODEL_NAME = "google/siglip-base-patch16-224"

devices = mesh_utils.create_device_mesh((jax.device_count(), 1))
mesh = Mesh(devices, ("data", "fsdp"))
jax.set_mesh(mesh)


@jax.jit
def create_model() -> SigLIP:
    """Create and shard SigLIP model.

    Returns:
        SigLIP: Sharded model.
    """
    model = SigLIP.from_pretrained(HF_MODEL_NAME, rngs=nnx.Rngs(0))
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    nnx.update(model, jax.lax.with_sharding_constraint(state, pspecs))
    return model


@jax.jit
def create_vision_model() -> SigLIPVisionModel:
    model = SigLIPVisionModel.from_pretrained(HF_MODEL_NAME, rngs=nnx.Rngs(0))
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    nnx.update(model, jax.lax.with_sharding_constraint(state, pspecs))
    return model


@jax.jit
def create_text_model() -> SigLIPTextModel:
    model = SigLIPTextModel.from_pretrained(HF_MODEL_NAME, rngs=nnx.Rngs(0))
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    nnx.update(model, jax.lax.with_sharding_constraint(state, pspecs))
    return model


def _vision_forward(model: SigLIPVisionModel, image: Float[Array, "batch height width channels"]) -> Float[Array, "batch hidden_size"]:
    return model(image)


def _text_forward(model: SigLIPTextModel, text: Int[Array, "batch seq_len"]) -> Float[Array, "batch hidden_size"]:
    return model(text)


def _siglip_forward(model: SigLIP, image: Float[Array, "batch height width channels"], text: Int[Array, "batch seq_len"]) -> Float[Array, "batch batch"]:
    return model(image, text)


vision_forward = nnx.jit(_vision_forward)
text_forward = nnx.jit(_text_forward)
siglip_forward = nnx.jit(_siglip_forward)


def test_siglip_vision_model() -> None:
    """Test SigLIPVisionModel standalone inference against HF reference.

    Returns:
        None
    """
    global mesh
    with mesh:
        vision_model = create_vision_model()

    image = Image.open("images/test_image.jpg")
    processor = AutoProcessor.from_pretrained(HF_MODEL_NAME)
    inputs = processor(images=image, return_tensors="pt")

    pytorch_model = SiglipVisionModel.from_pretrained(HF_MODEL_NAME)
    pytorch_model.eval()
    outputs = pytorch_model(**inputs)
    image_features_ref = outputs.pooler_output.detach().cpu().numpy()
    print(f"Vision Model - Reference shape: {image_features_ref.shape}")

    vision_model.eval()
    image_array: Float[Array, "batch height width channels"] = jnp.transpose(inputs["pixel_values"].detach().cpu().numpy(), axes=(0, 2, 3, 1))
    image_features_jimm = vision_forward(vision_model, image_array)

    print(f"Vision Model - Max absolute difference: {jnp.abs(image_features_jimm - image_features_ref).max()}")
    assert jnp.allclose(image_features_jimm, image_features_ref, atol=2e-2), f"Vision outputs don't match: max diff {jnp.abs(image_features_jimm - image_features_ref).max()}"


def test_siglip_text_model() -> None:
    """Test SigLIPTextModel standalone inference against HF reference.

    Returns:
        None
    """
    global mesh
    with mesh:
        text_model = create_text_model()

    processor = AutoProcessor.from_pretrained(HF_MODEL_NAME)
    text = ["a photo of a dog", "a photo of a cat"]
    inputs = processor(text=text, return_tensors="pt", padding="max_length")

    pytorch_text_model = SiglipTextModel.from_pretrained(HF_MODEL_NAME)
    pytorch_text_model.eval()
    outputs = pytorch_text_model(**inputs)
    text_features_ref = outputs.pooler_output.detach().cpu().numpy()
    print(f"Text Model - Reference shape: {text_features_ref.shape}")

    text_model.eval()
    text_array: Int[Array, "batch seq_len"] = inputs["input_ids"].detach().cpu().numpy()
    text_features_jimm = text_forward(text_model, text_array)

    print(f"Text Model - Max absolute difference: {jnp.abs(text_features_jimm - text_features_ref).max()}")
    assert jnp.allclose(text_features_jimm, text_features_ref, atol=2e-2), f"Text outputs don't match: max diff {jnp.abs(text_features_jimm - text_features_ref).max()}"


def test_siglip_inference() -> None:
    """Run SigLIP full model inference and compare to HF reference.

    Returns:
        None
    """
    global mesh
    with mesh:
        model = create_model()

    image = Image.open("images/test_image.jpg")
    processor = AutoProcessor.from_pretrained(HF_MODEL_NAME)
    inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt")

    pytorch_model = AutoModel.from_pretrained(HF_MODEL_NAME)
    pytorch_model.eval()
    outputs = pytorch_model(**inputs)
    logits_per_image_ref = outputs.logits_per_image.detach().cpu().numpy()

    model.eval()
    image_array = jnp.transpose(inputs["pixel_values"].detach().cpu().numpy(), axes=(0, 2, 3, 1))
    text_array = inputs["input_ids"].detach().cpu().numpy()
    logits_per_image_flax = siglip_forward(model, image_array, text_array)
    print(f"Full Model - Max absolute difference: {jnp.abs(logits_per_image_flax - logits_per_image_ref).max()}")
    assert jnp.allclose(logits_per_image_flax, logits_per_image_ref, atol=4e-2), f"Full model outputs don't match: max diff {jnp.abs(logits_per_image_flax - logits_per_image_ref).max()}"


def test_siglip_from_config() -> None:
    """Test SigLIP.from_config creates model with correct architecture.

    Returns:
        None
    """
    config = AutoConfig.from_pretrained(HF_MODEL_NAME).to_dict()
    model = SigLIP.from_config(config, rngs=nnx.Rngs(0))

    text_config = config["text_config"]
    vision_config = config["vision_config"]
    image = jnp.ones((1, vision_config["image_size"], vision_config["image_size"], 3))
    text = jnp.ones((2, text_config["max_position_embeddings"]), dtype=jnp.int32)
    output = model(image, text)
    assert output.shape == (1, 2)


def test_siglip_vision_model_from_config() -> None:
    """Test SigLIPVisionModel.from_config creates model with correct architecture.

    Returns:
        None
    """
    config = AutoConfig.from_pretrained(HF_MODEL_NAME).to_dict()
    model = SigLIPVisionModel.from_config(config, rngs=nnx.Rngs(0))

    vision_config = config["vision_config"]
    image = jnp.ones((1, vision_config["image_size"], vision_config["image_size"], 3))
    output = model(image)
    assert output.shape == (1, vision_config["hidden_size"])


def test_siglip_text_model_from_config() -> None:
    """Test SigLIPTextModel.from_config creates model with correct architecture.

    Returns:
        None
    """
    config = AutoConfig.from_pretrained(HF_MODEL_NAME).to_dict()
    model = SigLIPTextModel.from_config(config, rngs=nnx.Rngs(0))

    text_config = config["text_config"]
    text = jnp.ones((2, text_config["max_position_embeddings"]), dtype=jnp.int32)
    output = model(text, do_projection=True)
    assert output.shape == (2, text_config["hidden_size"])


@pytest.mark.parametrize("sharding", [NoSharding(), SigLIPSharding()], ids=["no_sharding", "siglip_sharding"])
def test_siglip_explicit_sharding(sharding: NoSharding | None) -> None:
    """Test SigLIP forward pass in JAX explicit sharding mode.

    Uses ``AxisType.Explicit`` so that ``jax.typeof(x).sharding`` is queryable
    at trace time inside ``@jax.jit``.  Both a fully-replicated (``NoSharding``)
    and the default FSDP (``SigLIPSharding``) configuration are tested.

    Args:
        sharding (NoSharding | None): Parameter sharding to use. ``None``
            selects the default ``SigLIPSharding``.

    Returns:
        None
    """

    n_devices = jax.device_count()
    explicit_devices = mesh_utils.create_device_mesh((n_devices, 1))
    explicit_mesh = Mesh(explicit_devices, ("data", "fsdp"), axis_types=(AxisType.Explicit, AxisType.Explicit))

    config = AutoConfig.from_pretrained(HF_MODEL_NAME).to_dict()
    text_config = config["text_config"]
    vision_config = config["vision_config"]
    traced_specs: dict[str, P] = {}

    @nnx.jit
    def forward(model: SigLIP, image: Float[Array, "batch h w c"], text: Int[Array, "batch seq"]) -> Float[Array, "batch batch"]:
        traced_specs["proj_kernel"] = jax.typeof(model.text_model.text_projection.kernel[...]).sharding.spec
        traced_specs["image"] = jax.typeof(image).sharding.spec
        traced_specs["text"] = jax.typeof(text).sharding.spec
        logits = model(image, text)
        traced_specs["output"] = jax.typeof(logits).sharding.spec
        return logits

    jax.set_mesh(explicit_mesh)
    try:
        model = SigLIP.from_config(config, rngs=nnx.Rngs(0), sharding=sharding if sharding is not None else SigLIPSharding())
        model.eval()
        image = jax.device_put(
            jnp.ones((n_devices, vision_config["image_size"], vision_config["image_size"], 3)),
            NamedSharding(explicit_mesh, P("data", None, None, None)),
        )
        text = jax.device_put(
            jnp.ones((n_devices, text_config["max_position_embeddings"]), dtype=jnp.int32),
            NamedSharding(explicit_mesh, P("data", None)),
        )
        out = jax.block_until_ready(forward(model, image, text))
    finally:
        jax.set_mesh(mesh)

    assert out.shape == (n_devices, n_devices)
    assert traced_specs["image"][0] == "data", f"image batch dim not sharded on 'data': {traced_specs['image']}"
    assert traced_specs["text"][0] == "data", f"text batch dim not sharded on 'data': {traced_specs['text']}"
    if n_devices > 1:
        assert traced_specs["output"] == P("data", None), f"unexpected logits sharding: {traced_specs['output']}"
    expected_proj = P(None, None) if isinstance(sharding, NoSharding) else P("fsdp", None)
    assert traced_specs["proj_kernel"] == expected_proj, f"unexpected proj_kernel sharding: {traced_specs['proj_kernel']}"


@pytest.mark.tokamax
@pytest.mark.parametrize("batch_size_per_device", [1, 2])
def test_siglip_tokamax_attention(batch_size_per_device: int, hf_model_name: str = HF_MODEL_NAME) -> None:
    """Test SigLIP with tokamax attention: correctness, latency, and peak HBM.

    Tests mosaic_tpu (falling back to xla_chunked on older TPUs like v4), plain
    xla_chunked, plain xla, and text-only mosaic_tpu backends. Parametrized over
    batch sizes to find the practical HBM ceiling on the current hardware.

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
    text_config = config["text_config"]
    vision_config = config["vision_config"]

    image_sharding = NamedSharding(mesh, P("data", None, None, None))
    text_sharding = NamedSharding(mesh, P("data", None))
    image = jax.device_put(
        jnp.ones((total_batch, vision_config["image_size"], vision_config["image_size"], 3)),
        image_sharding,
    )
    text = jax.device_put(
        jnp.ones((total_batch, text_config["max_position_embeddings"]), dtype=jnp.int32),
        text_sharding,
    )

    print(f"\nModel: {hf_model_name}  (vision_seq={vision_config['image_size'] ** 2 // vision_config['patch_size'] ** 2}, text_seq={text_config['max_position_embeddings']})")
    mosaic = jimm.make_tokamax_attention(["mosaic_tpu", "xla_chunked"])
    models = {
        "standard": SigLIP.from_config(config, rngs=nnx.Rngs(0)),
        "mosaic": SigLIP.from_config(config, rngs=nnx.Rngs(0), attention_fn=mosaic),
        "chunked": SigLIP.from_config(config, rngs=nnx.Rngs(0), attention_fn=jimm.make_tokamax_attention("xla_chunked")),
        "xla": SigLIP.from_config(config, rngs=nnx.Rngs(0), attention_fn=jimm.make_tokamax_attention("xla")),
        "vision_only": SigLIP.from_config(config, rngs=nnx.Rngs(0), vision_attention_fn=mosaic),
        "text_only": SigLIP.from_config(config, rngs=nnx.Rngs(0), text_attention_fn=mosaic),
    }
    for m in models.values():
        m.eval()

    @nnx.jit
    def forward(model: SigLIP, image: Float[Array, "batch height width channels"], text: Int[Array, "batch seq_len"]) -> Float[Array, "batch batch"]:
        return model(image, text)

    tuned = {k: jimm.autotuned_fn(forward, models[k], image, text, cache_dir=AUTOTUNE_CACHE_DIR) for k in models}
    outs = {k: jax.block_until_ready(tuned[k](models[k], image, text)) for k in models}
    ms = {k: bench(tuned[k], models[k], image, text) for k in models}
    peaks = {k: peak_hbm_mb(forward, models[k], image, text) for k in models}

    print(f"batch_size={total_batch}  ({batch_size_per_device} per device × {n_devices} devices)")
    ref = ms["standard"]
    for k, label in [
        ("standard", "Standard:                          "),
        ("mosaic", "Tokamax (mosaic_tpu→chunked):      "),
        ("chunked", "Tokamax (xla_chunked):             "),
        ("xla", "Tokamax (xla):                     "),
        ("vision_only", "Tokamax (vision-only mosaic_tpu):  "),
        ("text_only", "Tokamax (text-only mosaic_tpu):    "),
    ]:
        speed = f"  ({ref / ms[k]:.2f}x speed)" if k != "standard" else ""
        print(f"{label}{ms[k]:.2f} ms/fwd  peak HBM: {peaks[k]:.1f} MB{speed}")
    for k in ("mosaic", "chunked", "xla", "vision_only", "text_only"):
        print(f"max diff {k:<12} vs standard: {jnp.abs(outs[k] - outs['standard']).max():.2e}")

    for k in ("mosaic", "chunked", "xla", "vision_only", "text_only"):
        assert jnp.allclose(outs["standard"], outs[k], atol=1e-2), f"{k} outputs differ: {jnp.abs(outs['standard'] - outs[k]).max()}"
