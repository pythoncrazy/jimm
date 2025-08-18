import jax
import jax.numpy as jnp
from flax import nnx
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jaxtyping import Array, Float, Int
from PIL import Image
from transformers import AutoModel, AutoProcessor, SiglipTextModel, SiglipVisionModel

from jimm.models.siglip import SigLIP, SigLIPVisionModel

HF_MODEL_NAME = "google/siglip-base-patch16-256"

devices = mesh_utils.create_device_mesh((jax.device_count(),))
mesh = Mesh(devices, ("model",))


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


def test_siglip_inference() -> None:
    """Run SigLIP inference and compare to HF reference.

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

    pytorch_model = SiglipVisionModel.from_pretrained(HF_MODEL_NAME)
    pytorch_model.eval()
    outputs = pytorch_model(**inputs)
    image_features_ref = outputs.pooler_output.detach().cpu().numpy()
    print(image_features_ref.shape)

    vision_model.eval()
    image_array: Float[Array, "batch height width channels"] = jnp.transpose(inputs["pixel_values"].detach().cpu().numpy(), axes=(0, 2, 3, 1))
    image_features_jimm = nnx.jit(vision_model)(image_array)

    print(f"Max Image features absolute difference: {jnp.abs(image_features_jimm - image_features_ref).max()}")
    assert jnp.allclose(image_features_jimm, image_features_ref, atol=2e-2), f"Outputs don't match: {image_features_jimm} vs {image_features_ref}"

    # Test text encoder
    pytorch_text_model = SiglipTextModel.from_pretrained(HF_MODEL_NAME)
    pytorch_text_model.eval()

    text = ["a photo of a dog", "a photo of a cat"]
    inputs = processor(text=text, return_tensors="pt", padding="max_length")

    outputs = pytorch_text_model(**inputs)
    text_features_ref = outputs.pooler_output.detach().cpu().numpy()

    text_array: Int[Array, "batch seq_len"] = inputs["input_ids"].detach().cpu().numpy()
    text_features_jimm = nnx.jit(model.encode_text)(text_array)

    print(f"Max Text features absolute difference: {jnp.abs(text_features_jimm - text_features_ref).max()}")
    assert jnp.allclose(text_features_jimm, text_features_ref, atol=2e-2), f"Outputs don't match: {text_features_jimm} vs {text_features_ref}"
    inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt")

    pytorch_model = AutoModel.from_pretrained(HF_MODEL_NAME)
    pytorch_model.eval()
    outputs = pytorch_model(**inputs)
    logits_per_image_ref = outputs.logits_per_image.detach().cpu().numpy()

    model.eval()
    image_array = jnp.transpose(inputs["pixel_values"].detach().cpu().numpy(), axes=(0, 2, 3, 1))
    text_array = inputs["input_ids"].detach().cpu().numpy()
    logits_per_image_flax = nnx.jit(model)(image_array, text_array)
    print(f"Max absolute difference: {jnp.abs(logits_per_image_flax - logits_per_image_ref).max()}")
    assert jnp.allclose(logits_per_image_flax, logits_per_image_ref, atol=2e-2), f"Outputs don't match: {logits_per_image_flax} vs {logits_per_image_ref}"
