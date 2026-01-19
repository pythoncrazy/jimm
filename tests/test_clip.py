import jax
import jax.numpy as jnp
from flax import nnx
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jaxtyping import Array, Float, Int
from PIL import Image
from transformers import AutoConfig, AutoProcessor, CLIPModel
from transformers import CLIPTextModelWithProjection as HFCLIPTextModel
from transformers import CLIPVisionModel as HFCLIPVisionModel

from jimm import CLIP, CLIPTextModel, CLIPVisionModel, SplashAttentionConfig

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
    image_features_jimm = nnx.jit(vision_model, static_argnums=1)(image_array, do_projection=False)

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
    text_features_jimm = nnx.jit(text_model, static_argnums=1)(text_array, do_projection=True)

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
    logits_per_image_flax = nnx.jit(model)(image_array, text_array)
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


def test_clip_splash_attention() -> None:
    """Test CLIP with splash attention config produces same output.

    Returns:
        None
    """
    splash_config = SplashAttentionConfig(enabled=False)
    model_with_splash = CLIP(
        image_resolution=224,
        vision_layers=2,
        vision_hidden_size=64,
        vision_patch_size=16,
        context_length=77,
        vocab_size=49408,
        text_hidden_size=64,
        num_text_heads=4,
        num_text_layers=2,
        splash_attention_config=splash_config,
        rngs=nnx.Rngs(0),
    )
    model_without_splash = CLIP(
        image_resolution=224,
        vision_layers=2,
        vision_hidden_size=64,
        vision_patch_size=16,
        context_length=77,
        vocab_size=49408,
        text_hidden_size=64,
        num_text_heads=4,
        num_text_layers=2,
        rngs=nnx.Rngs(0),
    )
    image: Float[Array, "batch height width channels"] = jnp.ones((1, 224, 224, 3))
    text = jnp.ones((2, 77), dtype=jnp.int32)
    output_with_splash = nnx.jit(model_with_splash)(image, text)
    output_without_splash = nnx.jit(model_without_splash)(image, text)
    print(f"Splash attention - Max absolute difference: {jnp.abs(output_with_splash - output_without_splash).max()}")
    assert output_with_splash.shape == (1, 2)
    assert jnp.allclose(output_with_splash, output_without_splash, atol=1e-5)
