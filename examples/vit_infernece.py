import jax
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
    # Visualize sharding for specific encoder layers
    # Assuming model.encoder.layers is a list of TransformerEncoder modules
    # For "google/vit-base-patch32-384", num_layers is 12
    if hasattr(model, "encoder") and hasattr(model.encoder, "layers") and len(model.encoder.layers) > 0:
        num_encoder_layers = len(model.encoder.layers)

        print("Visualizing sharding for the first encoder layer (index 0):")
        jax.debug.visualize_array_sharding(model.encoder.layers[0])

        if num_encoder_layers > 1:
            middle_layer_idx = (num_encoder_layers -1) // 2 # A middle layer
            if middle_layer_idx != 0 : # Avoid re-printing if only 2 layers and middle is 0
                print(f"Visualizing sharding for a middle encoder layer (index {middle_layer_idx}):")
                jax.debug.visualize_array_sharding(model.encoder.layers[middle_layer_idx])
        
        if num_encoder_layers > 1: # Last layer
            last_layer_idx = num_encoder_layers - 1
            # Avoid re-printing if last is same as first or middle
            if last_layer_idx != 0 and last_layer_idx != middle_layer_idx :
                print(f"Visualizing sharding for the last encoder layer (index {last_layer_idx}):")
                jax.debug.visualize_array_sharding(model.encoder.layers[last_layer_idx])
    else:
        print("Could not find encoder layers to visualize.")

    logits_flax = nnx.jit(model)(x_eval)


print(f"Logits shape: {logits_flax.shape}")
predicted_class_idx = jnp.argmax(logits_flax, axis=-1)
print(f"Predicted class index: {predicted_class_idx[0]}")
