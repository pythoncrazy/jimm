import jax
import jax.numpy as jnp
import requests
from flax import nnx
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int
from PIL import Image
from transformers import CLIPProcessor

from jimm.models.clip import CLIP

HF_MODEL_NAME = "openai/clip-vit-base-patch32"
USE_PYTORCH = True

devices = mesh_utils.create_device_mesh((1, jax.device_count()))
mesh = Mesh(devices, ("batch", "model"))

model = CLIP.from_pretrained(HF_MODEL_NAME, use_pytorch=USE_PYTORCH, mesh=mesh)
processor = CLIPProcessor.from_pretrained(HF_MODEL_NAME)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

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
