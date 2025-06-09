import jax
import jax.numpy as jnp
import requests
from flax import nnx
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from PIL import Image
from transformers import ViTImageProcessor

from jimm.common.vit import VisionTransformer

HF_MODEL_NAME = "google/vit-large-patch16-384"
IMG_SIZE = 384
BATCH_SIZE = 128

mesh = Mesh(mesh_utils.create_device_mesh((1, 2)), ("batch", "model"))
model = VisionTransformer.from_pretrained(HF_MODEL_NAME, use_pytorch=True, mesh=mesh)
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

single_image = jnp.transpose(inputs["pixel_values"], axes=(0, 2, 3, 1))
x_eval = jnp.tile(single_image, (BATCH_SIZE, 1, 1, 1))

print(f"Input batch shape: {x_eval.shape}")

with mesh:
    x_eval_sharded = jax.device_put(x_eval, NamedSharding(mesh, P("batch")))
    logits_flax = nnx.jit(model)(x_eval_sharded)

print(f"Logits shape: {logits_flax.shape}")
predicted_class_idx = jnp.argmax(logits_flax, axis=-1)
print(f"Predicted class indices (first 10): {predicted_class_idx[:10]}")
print(f"All predictions are identical: {jnp.all(predicted_class_idx == predicted_class_idx[0])}")
