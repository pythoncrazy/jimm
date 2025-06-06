import jax.numpy as jnp
import requests
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

from src.vit import VisionTransformer


def test_vision_transformer_inference():
    HF_MODEL_NAME = "google/vit-base-patch16-224"
    SAFETENSORS_PATH = "weights/model-base-16-224.safetensors"

    model, inferred_img_size = VisionTransformer.from_pretrained(SAFETENSORS_PATH)

    url = "https://farm2.staticflickr.com/1152/1151216944_1525126615_z.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    processor = ViTImageProcessor.from_pretrained(HF_MODEL_NAME)

    inputs = processor(
        images=image,
        return_tensors="pt",
        size={"height": inferred_img_size, "width": inferred_img_size},
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
    print(f"Max absolute difference: {max_abs_diff}")
    assert max_abs_diff < 0.05
