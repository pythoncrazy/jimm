import io

import jax.numpy as jnp
import pytest
import requests
from flax import nnx
from PIL import Image
from transformers import AutoProcessor, CLIPModel

from jimm.models.clip import CLIP

HF_MODEL_NAME = "openai/clip-vit-large-patch14"


@pytest.mark.parametrize("use_pytorch", [False, True])
def test_clip_inference(use_pytorch):
    model = CLIP.from_pretrained(HF_MODEL_NAME, use_pytorch=use_pytorch)
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    response = requests.get(url)
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content))

    processor = AutoProcessor.from_pretrained(HF_MODEL_NAME)

    inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt")

    pytorch_model = CLIPModel.from_pretrained(HF_MODEL_NAME)
    pytorch_model.eval()
    outputs = pytorch_model(**inputs)
    logits_per_image_ref = outputs.logits_per_image.detach().cpu().numpy()

    model.eval()
    image_array = jnp.transpose(inputs["pixel_values"].detach().cpu().numpy(), axes=(0, 2, 3, 1))
    text_array = inputs["input_ids"].detach().cpu().numpy()
    logits_per_image_flax = nnx.jit(model)(image_array, text_array)
    print(f"Max absolute difference: {jnp.abs(logits_per_image_flax - logits_per_image_ref).max()}")
    assert jnp.allclose(logits_per_image_flax, logits_per_image_ref, atol=1e-1), f"Outputs don't match: {logits_per_image_flax} vs {logits_per_image_ref}"
