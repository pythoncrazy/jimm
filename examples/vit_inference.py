import jax
import jax.numpy as jnp
import requests
from flax import nnx
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from PIL import Image
from transformers import ViTImageProcessor
from jaxtyping import Array, Float, Int

from jimm.models.vit import VisionTransformer

HF_MODEL_NAME = "google/vit-large-patch16-384"
IMG_SIZE = 384
BATCH_SIZE = 128
NUM_BATCHES = 128
TOTAL_IMAGES = BATCH_SIZE * NUM_BATCHES

mesh = Mesh(mesh_utils.create_device_mesh((2, 1)), ("batch", "model"))
model = VisionTransformer.from_pretrained(HF_MODEL_NAME, use_pytorch=True, mesh=mesh, dtype=jnp.bfloat16)
model.eval()

url = "https://farm2.staticflickr.com/1152/1151216944_1525126615_z.jpg"
image = Image.open(requests.get(url, stream=True).raw)

processor = ViTImageProcessor.from_pretrained(HF_MODEL_NAME)

inputs = processor(
    images=image,
    return_tensors="np",
    size={"height": IMG_SIZE, "width": IMG_SIZE},
    do_resize=True,
)

pixel_values: Float[Array, f"1 3 {IMG_SIZE} {IMG_SIZE}"] = inputs["pixel_values"]
single_image: Float[Array, f"1 {IMG_SIZE} {IMG_SIZE} 3"] = jnp.transpose(pixel_values, axes=(0, 2, 3, 1))

print(f"Processing {NUM_BATCHES} batches of {BATCH_SIZE} images each")
print(f"Total images to process: {TOTAL_IMAGES}")

all_logits = []
all_predictions = []
forward = nnx.jit(model)  # Unfortunately, flax nnx always jits the model when this is called, so we should do it once and reuse it.
for batch_idx in range(NUM_BATCHES):
    print(f"\nProcessing batch {batch_idx + 1}/{NUM_BATCHES}")

    x_batch: Float[Array, f"{BATCH_SIZE} {IMG_SIZE} {IMG_SIZE} 3"] = jnp.tile(single_image, (BATCH_SIZE, 1, 1, 1))
    print(f"Batch shape: {x_batch.shape}")

    with mesh:
        x_batch_sharded = jax.device_put(x_batch, NamedSharding(mesh, P("batch", None, None, None)))

        logits_batch: Float[Array, f"{BATCH_SIZE} num_classes"] = forward(x_batch_sharded)

    print(f"Batch logits shape: {logits_batch.shape}")

    predicted_class_idx_batch: Int[Array, f"{BATCH_SIZE}"] = jnp.argmax(logits_batch, axis=-1)

    all_logits.append(logits_batch)
    all_predictions.append(predicted_class_idx_batch)

    print(f"Batch predictions (first 5): {predicted_class_idx_batch[:5]}")

combined_logits: Float[Array, f"{TOTAL_IMAGES} num_classes"] = jnp.concatenate(all_logits, axis=0)
combined_predictions: Int[Array, f"{TOTAL_IMAGES}"] = jnp.concatenate(all_predictions, axis=0)

print("\n=== Final Results ===")
print(f"Combined logits shape: {combined_logits.shape}")
print(f"Combined predictions shape: {combined_predictions.shape}")
print(f"Final predicted class indices (first 10): {combined_predictions[:10]}")
print(f"All predictions are identical: {jnp.all(combined_predictions == combined_predictions[0])}")
