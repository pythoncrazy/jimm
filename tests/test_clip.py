import jax.numpy as jnp
import pytest
import requests
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from jimm.models.clip import CLIP

HF_MODEL_NAME = "openai/clip-vit-large-patch14"


@pytest.mark.parametrize("use_pytorch", [False, True])
def test_clip_inference(use_pytorch):
    model = CLIP.from_pretrained(HF_MODEL_NAME, use_pytorch=use_pytorch)

    # Debug the model architecture
    # print(f"Testing CLIP with use_pytorch={use_pytorch}")
    # print(f"Vision width: {model.vision_width}")
    # print(f"Transformer width: {model.transformer_width}")
    # print(f"Vision model output dim: {model.vision_model.output_dim}")

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    processor = CLIPProcessor.from_pretrained(HF_MODEL_NAME)

    inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt")

    pytorch_model = CLIPModel.from_pretrained(HF_MODEL_NAME)
    pytorch_model.eval()
    outputs = pytorch_model(**inputs)
    logits_per_image_ref = outputs.logits_per_image.detach().cpu().numpy()

    model.eval()
    image_array = jnp.transpose(inputs["pixel_values"].detach().cpu().numpy(), axes=(0, 2, 3, 1))
    text_array = inputs["input_ids"].detach().cpu().numpy()

    # print(f"Image array shape: {image_array.shape}")
    # print(f"Text array shape: {text_array.shape}")

    logits_per_image_flax = model(image_array, text_array)

    # print(f"Reference logits shape: {logits_per_image_ref.shape}")
    # print(f"Our logits shape: {logits_per_image_flax.shape}")

    print(f"Max absolute difference: {jnp.abs(logits_per_image_flax - logits_per_image_ref).max()}")
    assert jnp.allclose(logits_per_image_flax, logits_per_image_ref, atol=1e-1), f"Outputs don't match: {logits_per_image_flax} vs {logits_per_image_ref}"
