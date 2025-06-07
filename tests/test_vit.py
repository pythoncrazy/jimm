import jax.numpy as jnp
import pytest
import requests
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

from jimm.vit import VisionTransformer

HF_MODEL_NAME = "google/vit-base-patch16-224"
SAFETENSORS_PATH = "weights/model-base-16-224.safetensors"
# The image size for "google/vit-base-patch16-224" is 224x224
IMG_SIZE = 224


@pytest.mark.parametrize("model_source", [SAFETENSORS_PATH, HF_MODEL_NAME])
def test_vision_transformer_inference(model_source):
    model = VisionTransformer.from_pretrained(model_source)

    url = "https://farm2.staticflickr.com/1152/1151216944_1525126615_z.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    processor = ViTImageProcessor.from_pretrained(HF_MODEL_NAME)

    inputs = processor(
        images=image,
        return_tensors="pt",
        size={"height": IMG_SIZE, "width": IMG_SIZE},
        do_resize=True,
    )

    pytorch_model = ViTForImageClassification.from_pretrained(HF_MODEL_NAME)
    pytorch_model.eval()
    outputs = pytorch_model(**inputs)
    logits_ref = outputs.logits.detach().cpu().numpy()

    model.eval()
    x_eval = jnp.transpose(inputs["pixel_values"].detach().cpu().numpy(), axes=(0, 2, 3, 1))
    logits_flax = model(x_eval)

    max_abs_diff = jnp.abs(logits_flax - logits_ref).max()
    print(f"Testing with model_source: {model_source}")
    print(f"Max absolute difference: {max_abs_diff}")
    assert max_abs_diff < 0.05
