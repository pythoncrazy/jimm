import jax
import jax.numpy as jnp
from flax import nnx
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jaxtyping import Array, Float
from PIL import Image
from transformers import AutoConfig, ViTForImageClassification, ViTImageProcessor

from jimm import SplashAttentionConfig, VisionTransformer

HF_MODEL_NAME = "google/vit-base-patch16-224"

devices = mesh_utils.create_device_mesh((jax.device_count(), 1))
mesh = Mesh(devices, ("data", "fsdp"))
jax.set_mesh(mesh)


@nnx.jit
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
    logits_flax = nnx.jit(model)(x_eval)
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


def test_vision_transformer_splash_attention() -> None:
    """Test VisionTransformer with splash attention config loads from HuggingFace and produces same output.

    Returns:
        None
    """
    splash_config = SplashAttentionConfig(enabled=True)
    model_with_splash = VisionTransformer.from_pretrained(
        HF_MODEL_NAME,
        splash_attention_config=splash_config,
        rngs=nnx.Rngs(0),
    )
    model_without_splash = VisionTransformer.from_pretrained(
        HF_MODEL_NAME,
        rngs=nnx.Rngs(0),
    )

    image = Image.open("images/test_image.jpg")
    processor = ViTImageProcessor.from_pretrained(HF_MODEL_NAME)
    inputs = processor(images=image, return_tensors="pt")

    model_with_splash.eval()
    model_without_splash.eval()

    x: Float[Array, "batch height width channels"] = jnp.transpose(inputs["pixel_values"].detach().cpu().numpy(), axes=(0, 2, 3, 1))
    output_with_splash = nnx.jit(model_with_splash)(x)
    output_without_splash = nnx.jit(model_without_splash)(x)
    print(f"Splash attention - Max absolute difference: {jnp.abs(output_with_splash - output_without_splash).max()}")
    assert output_with_splash.shape == output_without_splash.shape
    assert jnp.allclose(output_with_splash, output_without_splash, atol=1e-5)
