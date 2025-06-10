import jax.numpy as jnp
import requests
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

from jimm.models.vit import VisionTransformer

HF_MODEL_NAME = "google/vit-base-patch32-384"
IMG_SIZE = 384


def test_vision_transformer_pytorch_loading():
    model = VisionTransformer.from_pretrained(HF_MODEL_NAME, use_pytorch=True)

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
    print(f"Testing with model_source: {HF_MODEL_NAME}")
    print(f"Max absolute difference: {max_abs_diff}")
    assert max_abs_diff < 0.05
