import jax.numpy as jnp
import requests
from flax import nnx
from PIL import Image
from transformers import AutoModel, SiglipProcessor, SiglipTextModel, SiglipVisionModel

from jimm.models.siglip import SigLIP

HF_MODEL_NAME = "google/siglip-base-patch16-256"


def test_siglip_inference():
    """
    Test SigLIP vision model inference against the Hugging Face implementation.
    """
    model = SigLIP.from_pretrained(HF_MODEL_NAME)

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    processor = SiglipProcessor.from_pretrained(HF_MODEL_NAME)
    inputs = processor(images=image, return_tensors="pt")

    pytorch_model = SiglipVisionModel.from_pretrained(HF_MODEL_NAME)
    pytorch_model.eval()
    outputs = pytorch_model(**inputs)
    image_features_ref = outputs.pooler_output.detach().cpu().numpy()
    print(image_features_ref.shape)

    model.eval()
    image_array = jnp.transpose(inputs["pixel_values"].detach().cpu().numpy(), axes=(0, 2, 3, 1))

    image_features_jimm = nnx.jit(model.encode_image)(image_array)

    print(f"Max Image features absolute difference: {jnp.abs(image_features_jimm - image_features_ref).max()}")
    assert jnp.allclose(image_features_jimm, image_features_ref, atol=1e-2), f"Outputs don't match: {image_features_jimm} vs {image_features_ref}"

    # Test text encoder
    pytorch_text_model = SiglipTextModel.from_pretrained(HF_MODEL_NAME)
    pytorch_text_model.eval()

    text = ["a photo of a dog", "a photo of a cat"]
    inputs = processor(text=text, return_tensors="pt", padding="max_length")

    outputs = pytorch_text_model(**inputs)
    text_features_ref = outputs.pooler_output.detach().cpu().numpy()

    text_array = inputs["input_ids"].detach().cpu().numpy()
    text_features_jimm = nnx.jit(model.encode_text)(text_array)

    print(f"Max Text features absolute difference: {jnp.abs(text_features_jimm - text_features_ref).max()}")
    assert jnp.allclose(text_features_jimm, text_features_ref, atol=1e-2), f"Outputs don't match: {text_features_jimm} vs {text_features_ref}"

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt")

    pytorch_model = AutoModel.from_pretrained(HF_MODEL_NAME)
    pytorch_model.eval()
    outputs = pytorch_model(**inputs)
    logits_per_image_ref = outputs.logits_per_image.detach().cpu().numpy()

    model.eval()
    image_array = jnp.transpose(inputs["pixel_values"].detach().cpu().numpy(), axes=(0, 2, 3, 1))
    text_array = inputs["input_ids"].detach().cpu().numpy()
    logits_per_image_flax = nnx.jit(model)(image_array, text_array)
    print(f"Max absolute difference: {jnp.abs(logits_per_image_flax - logits_per_image_ref).max()}")
    assert jnp.allclose(logits_per_image_flax, logits_per_image_ref, atol=1e-2), f"Outputs don't match: {logits_per_image_flax} vs {logits_per_image_ref}"
