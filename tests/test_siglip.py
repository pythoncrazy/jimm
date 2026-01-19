import jax
import jax.numpy as jnp
from flax import nnx
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jaxtyping import Array, Float, Int
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoProcessor, SiglipTextModel, SiglipVisionModel

from jimm import SigLIP, SigLIPTextModel, SigLIPVisionModel, SplashAttentionConfig

HF_MODEL_NAME = "google/siglip-base-patch16-256"

devices = mesh_utils.create_device_mesh((jax.device_count(), 1))
mesh = Mesh(devices, ("data", "fsdp"))
jax.set_mesh(mesh)


@nnx.jit
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


@nnx.jit
def create_vision_model() -> SigLIPVisionModel:
    model = SigLIPVisionModel.from_pretrained(HF_MODEL_NAME, rngs=nnx.Rngs(0))
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    nnx.update(model, jax.lax.with_sharding_constraint(state, pspecs))
    return model


@nnx.jit
def create_text_model() -> SigLIPTextModel:
    model = SigLIPTextModel.from_pretrained(HF_MODEL_NAME, rngs=nnx.Rngs(0))
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    nnx.update(model, jax.lax.with_sharding_constraint(state, pspecs))
    return model


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
    image_features_jimm = jax.jit(vision_model)(image_array)

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
    text_features_jimm = jax.jit(text_model)(text_array)

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
    logits_per_image_flax = nnx.jit(model)(image_array, text_array)
    print(f"Full Model - Max absolute difference: {jnp.abs(logits_per_image_flax - logits_per_image_ref).max()}")
    assert jnp.allclose(logits_per_image_flax, logits_per_image_ref, atol=3e-2), f"Full model outputs don't match: max diff {jnp.abs(logits_per_image_flax - logits_per_image_ref).max()}"


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


def test_siglip_splash_attention() -> None:
    """Test SigLIP with splash attention config loads from HuggingFace and produces same output.

    Returns:
        None
    """
    splash_config = SplashAttentionConfig(enabled=True)
    model_with_splash = SigLIP.from_pretrained(
        HF_MODEL_NAME,
        splash_attention_config=splash_config,
        rngs=nnx.Rngs(0),
    )
    model_without_splash = SigLIP.from_pretrained(
        HF_MODEL_NAME,
        rngs=nnx.Rngs(0),
    )

    image = Image.open("images/test_image.jpg")
    processor = AutoProcessor.from_pretrained(HF_MODEL_NAME)
    inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt")

    image_array: Float[Array, "batch height width channels"] = jnp.transpose(inputs["pixel_values"].detach().cpu().numpy(), axes=(0, 2, 3, 1))
    text_array: Int[Array, "batch seq_len"] = inputs["input_ids"].detach().cpu().numpy()

    output_with_splash = nnx.jit(model_with_splash)(image_array, text_array)
    output_without_splash = nnx.jit(model_without_splash)(image_array, text_array)
    print(f"Splash attention - Max absolute difference: {jnp.abs(output_with_splash - output_without_splash).max()}")
    assert output_with_splash.shape == output_without_splash.shape
    assert jnp.allclose(output_with_splash, output_without_splash, atol=1e-5)
