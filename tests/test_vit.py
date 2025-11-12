import jax
import jax.numpy as jnp
from flax import nnx
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jaxtyping import Array, Float
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

from jimm.models.vit import VisionTransformer

HF_MODEL_NAME = "google/vit-base-patch16-224"

devices = mesh_utils.create_device_mesh((jax.device_count(),))
mesh = Mesh(devices, ("fsdp",))


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


test_vision_transformer_inference()
