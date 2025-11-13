import jax
import jax.numpy as jnp
from flax import nnx
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jaxtyping import Array, Float, Int
from PIL import Image
from transformers import AutoProcessor, CLIPModel
from transformers import CLIPVisionModel as HFCLIPVisionModel

from jimm.models.clip import CLIP, CLIPVisionModel

HF_MODEL_NAME = "openai/clip-vit-large-patch14"

devices = mesh_utils.create_device_mesh((jax.device_count(), 1))
mesh = Mesh(devices, ("data", "fsdp"))
jax.set_mesh(mesh)


@jax.jit
def create_model() -> CLIP:
    model = CLIP.from_pretrained(HF_MODEL_NAME, rngs=nnx.Rngs(0))
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    nnx.update(model, jax.lax.with_sharding_constraint(state, pspecs))
    return model


@jax.jit
def create_vision_model() -> CLIPVisionModel:
    model = CLIPVisionModel.from_pretrained(HF_MODEL_NAME, rngs=nnx.Rngs(0))
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    nnx.update(model, jax.lax.with_sharding_constraint(state, pspecs))
    return model


def test_clip_inference() -> None:
    """Run CLIP inference and compare to HF reference.

    Args:
        use_pytorch (bool): Whether to load PyTorch weights.

    Returns:
        None
    """
    global mesh
    with mesh:
        model = create_model()
        vision_model = create_vision_model()
    image = Image.open("images/test_image.jpg")

    processor = AutoProcessor.from_pretrained(HF_MODEL_NAME)
    inputs = processor(images=image, return_tensors="pt")

    # Test Vision Model
    pytorch_model = HFCLIPVisionModel.from_pretrained(HF_MODEL_NAME)
    pytorch_model.eval()
    outputs = pytorch_model(**inputs)
    image_features_ref = outputs.pooler_output.detach().cpu().numpy()
    print(image_features_ref.shape)

    vision_model.eval()
    image_array: Float[Array, "batch height width channels"] = jnp.transpose(inputs["pixel_values"].detach().cpu().numpy(), axes=(0, 2, 3, 1))
    image_features_jimm = nnx.jit(vision_model, static_argnums=1)(image_array, do_projection=False)

    print(f"Max Image features absolute difference: {jnp.abs(image_features_jimm - image_features_ref).max()}")
    assert jnp.allclose(image_features_jimm, image_features_ref, atol=1e-1), f"Outputs don't match: {image_features_jimm} vs {image_features_ref}"

    # Test text encoder doesn't really matter too much, and I'm too lazy to do it
    # TODO: add test for text encoder alone
    inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt")
    # Test overall model
    pytorch_model = CLIPModel.from_pretrained(HF_MODEL_NAME)
    pytorch_model.eval()
    outputs = pytorch_model(**inputs)
    logits_per_image_ref = outputs.logits_per_image.detach().cpu().numpy()

    model.eval()
    image_array: Float[Array, "batch height width channels"] = jnp.transpose(inputs["pixel_values"].detach().cpu().numpy(), axes=(0, 2, 3, 1))
    text_array: Int[Array, "batch seq_len"] = inputs["input_ids"].detach().cpu().numpy()
    logits_per_image_flax = nnx.jit(model)(image_array, text_array)
    print(f"Max absolute difference: {jnp.abs(logits_per_image_flax - logits_per_image_ref).max()}")
    assert jnp.allclose(logits_per_image_flax, logits_per_image_ref, atol=1e-1), f"Outputs don't match: {logits_per_image_flax} vs {logits_per_image_ref}"
