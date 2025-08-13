import io

import jax
import jax.numpy as jnp
import requests
from flax import nnx
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int
from PIL import Image
from transformers import AutoProcessor

from jimm.models import CLIP

HF_MODEL_NAME = "geolocal/StreetCLIP"
USE_PYTORCH = True

devices = mesh_utils.create_device_mesh((1, jax.device_count()))
mesh = Mesh(devices, ("batch", "model"))


@nnx.jit
def create_sharded_model() -> CLIP:
    """Create and shard the CLIP model and optimizer following FSDP pattern.

    Returns:
        Tuple[CLIP, nnx.Optimizer]: Sharded model and optimizer
    """
    model = CLIP.from_pretrained(HF_MODEL_NAME, use_pytorch=USE_PYTORCH, use_gradient_checkpointing=True, rngs=nnx.Rngs(0))
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state)
    return model


with mesh:
    model = create_sharded_model()


processor = AutoProcessor.from_pretrained(HF_MODEL_NAME)


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
response = requests.get(url)
response.raise_for_status()
image = Image.open(io.BytesIO(response.content))

text_prompts = [
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a person",
    "a photo of a building",
    "a photo of food",
    "a photo of a landscape",
]

inputs = processor(text=text_prompts, images=image, return_tensors="pt", padding=True)

image_array: Float[Array, "batch height width channels"] = jnp.transpose(inputs["pixel_values"].detach().cpu().numpy(), axes=(0, 2, 3, 1))
text_array: Int[Array, "batch seq_len"] = inputs["input_ids"].detach().cpu().numpy()

with mesh:
    image_array_sharded = jax.device_put(image_array, NamedSharding(mesh, P("batch", None, None, None)))
    text_array_sharded = jax.device_put(text_array, NamedSharding(mesh, P("batch", None)))

    logits: Float[Array, "batch batch"] = nnx.jit(model)(image_array_sharded, text_array_sharded)

similarity_scores: Float[Array, " batch "] = logits[0]
softmax_scores: Float[Array, " batch "] = jnp.exp(similarity_scores) / jnp.sum(jnp.exp(similarity_scores))

indices: Int[Array, " batch "] = jnp.argsort(similarity_scores, axis=-1)[::-1]
sorted_scores: Float[Array, " batch "] = similarity_scores[indices]
sorted_softmax: Float[Array, " batch "] = softmax_scores[indices]
sorted_prompts = [text_prompts[i] for i in indices]

print("\nResults (sorted by similarity):")
print(f"{'Text Prompt':<25} | {'Score':<10} | {'Probability':<10}")
print("-" * 50)
for prompt, score, prob in zip(sorted_prompts, sorted_scores, sorted_softmax):
    print(f"{prompt[:25]:<25} | {score:.4f}     | {prob:.4f}")

print("\nBest match:", sorted_prompts[0])
