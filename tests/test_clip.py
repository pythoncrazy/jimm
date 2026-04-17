import jax
import jax.numpy as jnp
import jimm
import pytest
from flax import nnx
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P
from jaxtyping import Array, Float, Int
from PIL import Image
from transformers import AutoConfig, AutoProcessor, CLIPModel
from transformers import CLIPTextModelWithProjection as HFCLIPTextModel
from transformers import CLIPVisionModel as HFCLIPVisionModel

from benchmark_utils import AUTOTUNE_CACHE_DIR, bench, peak_hbm_mb
from jimm import CLIP, CLIPTextModel, CLIPVisionModel

HF_MODEL_NAME = "openai/clip-vit-large-patch14"

devices = mesh_utils.create_device_mesh((jax.device_count(), 1))
mesh = Mesh(devices, ("data", "fsdp"))
jax.set_mesh(mesh)


@nnx.jit
def create_model() -> CLIP:
    model = CLIP.from_pretrained(HF_MODEL_NAME, rngs=nnx.Rngs(0))
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    nnx.update(model, jax.lax.with_sharding_constraint(state, pspecs))
    return model


@nnx.jit
def create_vision_model() -> CLIPVisionModel:
    model = CLIPVisionModel.from_pretrained(HF_MODEL_NAME, rngs=nnx.Rngs(0))
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    nnx.update(model, jax.lax.with_sharding_constraint(state, pspecs))
    return model


@nnx.jit
def create_text_model() -> CLIPTextModel:
    model = CLIPTextModel.from_pretrained(HF_MODEL_NAME, rngs=nnx.Rngs(0))
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    nnx.update(model, jax.lax.with_sharding_constraint(state, pspecs))
    return model


def _vision_forward(model: CLIPVisionModel, image: Float[Array, "batch height width channels"], do_projection: bool) -> Float[Array, "batch vision_hidden_size_or_projection_dim"]:
    return model(image, do_projection=do_projection)


def _text_forward(model: CLIPTextModel, text: Int[Array, "batch seq_len"], do_projection: bool) -> Float[Array, "batch text_hidden_size"]:
    return model(text, do_projection=do_projection)


def _clip_forward(model: CLIP, image: Float[Array, "batch height width channels"], text: Int[Array, "batch seq_len"]) -> Float[Array, "batch batch"]:
    return model(image, text)


vision_forward = nnx.jit(_vision_forward, static_argnums=2)
text_forward = nnx.jit(_text_forward, static_argnums=2)
clip_forward = nnx.jit(_clip_forward)


def test_clip_vision_model() -> None:
    """Test CLIPVisionModel standalone inference against HF reference.

    Returns:
        None
    """
    global mesh
    with mesh:
        vision_model = create_vision_model()

    image = Image.open("images/test_image.jpg")
    processor = AutoProcessor.from_pretrained(HF_MODEL_NAME)
    inputs = processor(images=image, return_tensors="pt")

    pytorch_model = HFCLIPVisionModel.from_pretrained(HF_MODEL_NAME)
    pytorch_model.eval()
    outputs = pytorch_model(**inputs)
    image_features_ref = outputs.pooler_output.detach().cpu().numpy()
    print(f"Vision Model - Reference shape: {image_features_ref.shape}")

    vision_model.eval()
    image_array: Float[Array, "batch height width channels"] = jnp.transpose(inputs["pixel_values"].detach().cpu().numpy(), axes=(0, 2, 3, 1))
    image_features_jimm = vision_forward(vision_model, image_array, False)

    print(f"Vision Model - Max absolute difference: {jnp.abs(image_features_jimm - image_features_ref).max()}")
    assert jnp.allclose(image_features_jimm, image_features_ref, atol=1e-1), f"Vision outputs don't match: max diff {jnp.abs(image_features_jimm - image_features_ref).max()}"


def test_clip_text_model() -> None:
    """Test CLIPTextModel standalone inference against HF reference.

    Returns:
        None
    """
    global mesh
    with mesh:
        text_model = create_text_model()

    processor = AutoProcessor.from_pretrained(HF_MODEL_NAME)
    text = ["a photo of a cat", "a photo of a dog"]
    inputs = processor(text=text, return_tensors="pt", padding="max_length")

    pytorch_model = HFCLIPTextModel.from_pretrained(HF_MODEL_NAME)
    pytorch_model.eval()
    outputs = pytorch_model(**inputs)
    text_features_ref = outputs.text_embeds.detach().cpu().numpy()
    print(f"Text Model - Reference shape: {text_features_ref.shape}")

    text_model.eval()
    text_array: Int[Array, "batch seq_len"] = inputs["input_ids"].detach().cpu().numpy()
    text_features_jimm = text_forward(text_model, text_array, True)

    print(f"Text Model - Max absolute difference: {jnp.abs(text_features_jimm - text_features_ref).max()}")
    assert jnp.allclose(text_features_jimm, text_features_ref, atol=1e-1), f"Text outputs don't match: max diff {jnp.abs(text_features_jimm - text_features_ref).max()}"


def test_clip_inference() -> None:
    """Run CLIP full model inference and compare to HF reference.

    Returns:
        None
    """
    global mesh
    with mesh:
        model = create_model()

    image = Image.open("images/test_image.jpg")

    processor = AutoProcessor.from_pretrained(HF_MODEL_NAME)
    inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt")
    pytorch_model = CLIPModel.from_pretrained(HF_MODEL_NAME)
    pytorch_model.eval()
    outputs = pytorch_model(**inputs)
    logits_per_image_ref = outputs.logits_per_image.detach().cpu().numpy()

    model.eval()
    image_array: Float[Array, "batch height width channels"] = jnp.transpose(inputs["pixel_values"].detach().cpu().numpy(), axes=(0, 2, 3, 1))
    text_array: Int[Array, "batch seq_len"] = inputs["input_ids"].detach().cpu().numpy()
    logits_per_image_flax = clip_forward(model, image_array, text_array)
    print(f"Full Model - Max absolute difference: {jnp.abs(logits_per_image_flax - logits_per_image_ref).max()}")
    assert jnp.allclose(logits_per_image_flax, logits_per_image_ref, atol=1e-1), f"Full model outputs don't match: max diff {jnp.abs(logits_per_image_flax - logits_per_image_ref).max()}"


def test_clip_from_config() -> None:
    """Test CLIP.from_config creates model with correct architecture.

    Returns:
        None
    """
    config = AutoConfig.from_pretrained(HF_MODEL_NAME).to_dict()
    model = CLIP.from_config(config, rngs=nnx.Rngs(0))

    text_config = config["text_config"]
    vision_config = config["vision_config"]
    image = jnp.ones((1, vision_config["image_size"], vision_config["image_size"], 3))
    text = jnp.ones((2, text_config["max_position_embeddings"]), dtype=jnp.int32)
    output = model(image, text)
    assert output.shape == (1, 2)


def test_clip_vision_model_from_config() -> None:
    """Test CLIPVisionModel.from_config creates model with correct architecture.

    Returns:
        None
    """
    config = AutoConfig.from_pretrained(HF_MODEL_NAME).to_dict()
    model = CLIPVisionModel.from_config(config, rngs=nnx.Rngs(0))

    text_config = config["text_config"]
    vision_config = config["vision_config"]
    image = jnp.ones((1, vision_config["image_size"], vision_config["image_size"], 3))
    output = model(image, do_projection=True)
    assert output.shape == (1, text_config["hidden_size"])

    output_no_proj = model(image, do_projection=False)
    assert output_no_proj.shape == (1, vision_config["hidden_size"])


def test_clip_text_model_from_config() -> None:
    """Test CLIPTextModel.from_config creates model with correct architecture.

    Returns:
        None
    """
    config = AutoConfig.from_pretrained(HF_MODEL_NAME).to_dict()
    model = CLIPTextModel.from_config(config, rngs=nnx.Rngs(0))

    text_config = config["text_config"]
    text = jnp.ones((2, text_config["max_position_embeddings"]), dtype=jnp.int32)
    output = model(text, do_projection=True)
    assert output.shape == (2, text_config["hidden_size"])



@pytest.mark.parametrize("batch_size_per_device", [8, 16, 32, 64])
def test_clip_tokamax_attention(batch_size_per_device: int, hf_model_name: str = HF_MODEL_NAME) -> None:
    """Test CLIP with tokamax attention: correctness, latency, and peak HBM vs standard attention.

    Tests mosaic_tpu (falling back to xla_chunked on older TPUs like v4), plain
    xla_chunked, and plain xla backends. Parametrized over batch sizes to find
    the practical HBM ceiling on the current hardware.

    Args:
        batch_size_per_device (int): Images / text sequences per device.
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

    # Pre-shard inputs along the batch dimension so all models see the same data layout.
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

    print(f"\nModel: {hf_model_name}")
    mosaic = jimm.make_tokamax_attention(["mosaic_tpu", "xla_chunked"])
    model_standard  = CLIP.from_config(config, rngs=nnx.Rngs(0))
    model_mosaic    = CLIP.from_config(config, rngs=nnx.Rngs(0), attention_fn=mosaic)
    model_chunked   = CLIP.from_config(config, rngs=nnx.Rngs(0), attention_fn=jimm.make_tokamax_attention("xla_chunked"))
    model_xla       = CLIP.from_config(config, rngs=nnx.Rngs(0), attention_fn=jimm.make_tokamax_attention("xla"))
    model_text_only = CLIP.from_config(config, rngs=nnx.Rngs(0), text_attention_fn=mosaic)

    model_standard.eval()
    model_mosaic.eval()
    model_chunked.eval()
    model_xla.eval()
    model_text_only.eval()

    @nnx.jit
    def forward(model: CLIP, image: Float[Array, "batch height width channels"], text: Int[Array, "batch seq_len"]) -> Float[Array, "batch batch"]:
        return model(image, text)

    models = {
        "standard":  model_standard,
        "mosaic":    model_mosaic,
        "chunked":   model_chunked,
        "xla":       model_xla,
        "text_only": model_text_only,
    }
    tuned = {k: jimm.autotuned_fn(forward, models[k], image, text, cache_dir=AUTOTUNE_CACHE_DIR) for k in models}
    outs  = {k: jax.block_until_ready(tuned[k](models[k], image, text)) for k in models}
    ms    = {k: bench(tuned[k], models[k], image, text)      for k in models}
    peaks = {k: peak_hbm_mb(forward, models[k], image, text) for k in models}

    print(f"batch_size={total_batch}  ({batch_size_per_device} per device × {n_devices} devices)")
    ref = ms["standard"]
    for k, label in [
        ("standard",  "Standard:                        "),
        ("mosaic",    "Tokamax (mosaic_tpu→chunked):    "),
        ("chunked",   "Tokamax (xla_chunked):           "),
        ("xla",       "Tokamax (xla):                   "),
        ("text_only", "Tokamax (text-only mosaic_tpu):  "),
    ]:
        speed = f"  ({ref / ms[k]:.2f}x speed)" if k != "standard" else ""
        print(f"{label}{ms[k]:.2f} ms/fwd  peak HBM: {peaks[k]:.1f} MB{speed}")
    for k in ("mosaic", "chunked", "xla", "text_only"):
        print(f"max diff {k:<10} vs standard: {jnp.abs(outs[k] - outs['standard']).max():.2e}")

    for k in ("mosaic", "chunked", "xla", "text_only"):
        assert jnp.allclose(outs["standard"], outs[k], atol=1e-2), f"{k} outputs differ: {jnp.abs(outs['standard'] - outs[k]).max()}"


def test_clip_autotune(hf_model_name: str = HF_MODEL_NAME) -> None:
    """Autotune tokamax ops for a CLIP model, caching results to tests/tokamax_cache/.

    Uses ``jimm.autotuned_fn`` which calls ``jimm.cached_autotune`` under the
    hood: on first run it benchmarks all kernel configs and writes a JSON cache
    file keyed by op shapes + device kind; on subsequent runs it loads the
    cache and skips benchmarking entirely.

    Args:
        hf_model_name (str): HuggingFace model name. Defaults to HF_MODEL_NAME.

    Returns:
        None
    """
    cache_dir = AUTOTUNE_CACHE_DIR

    config = AutoConfig.from_pretrained(hf_model_name).to_dict()
    text_config = config["text_config"]
    vision_config = config["vision_config"]

    model = CLIP.from_config(
        config,
        rngs=nnx.Rngs(0),
        attention_fn=jimm.make_tokamax_attention(["mosaic_tpu", "xla_chunked"]),
    )
    model.eval()

    image = jnp.ones((1, vision_config["image_size"], vision_config["image_size"], 3))
    text = jnp.ones((1, text_config["max_position_embeddings"]), dtype=jnp.int32)

    @nnx.jit
    def forward(model: CLIP, image: Float[Array, "batch height width channels"], text: Int[Array, "batch seq_len"]) -> Float[Array, "batch batch"]:
        return model(image, text)

    print(f"\nAutotuning {hf_model_name}  (cache_dir={cache_dir})")
    tuned_forward = jimm.autotuned_fn(forward, model, image, text, cache_dir=cache_dir)

    out = jax.block_until_ready(tuned_forward(model, image, text))
    assert out.shape == (1, 1)
