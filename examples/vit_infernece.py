import jax.numpy as jnp
import requests
from flax import nnx
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from PIL import Image
from transformers import ViTImageProcessor

from jimm.common.vit import VisionTransformer

HF_MODEL_NAME = "google/vit-base-patch32-384"
IMG_SIZE = 384
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

x_eval = jnp.transpose(inputs["pixel_values"], axes=(0, 2, 3, 1))
with mesh:
    logits_flax = nnx.jit(model)(x_eval)


print(f"Logits shape: {logits_flax.shape}")
predicted_class_idx = jnp.argmax(logits_flax, axis=-1)
print(f"Predicted class index: {predicted_class_idx[0]}")
