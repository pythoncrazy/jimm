import jax.numpy as jnp
import requests
from PIL import Image
from transformers import ViTImageProcessor

from jimm.vit import VisionTransformer

HF_MODEL_NAME = "google/vit-base-patch16-224"
SAFETENSORS_PATH = "weights/model-base-16-224.safetensors"
IMG_SIZE = 224

model = VisionTransformer.from_pretrained(SAFETENSORS_PATH)
model.eval()

url = "https://farm2.staticflickr.com/1152/1151216944_1525126615_z.jpg"
image = Image.open(requests.get(url, stream=True).raw)

processor = ViTImageProcessor.from_pretrained(HF_MODEL_NAME)

inputs = processor(
    images=image,
    return_tensors="pt",
    size={"height": IMG_SIZE, "width": IMG_SIZE},
    do_resize=True,
)

x_eval = jnp.transpose(inputs["pixel_values"].detach().cpu().numpy(), axes=(0, 2, 3, 1))
logits_flax = model(x_eval)

print("Logits from JAX/Flax model:")
print(logits_flax)
print(f"Logits shape: {logits_flax.shape}")
predicted_class_idx = jnp.argmax(logits_flax, axis=-1)
print(f"Predicted class index: {predicted_class_idx[0]}")
