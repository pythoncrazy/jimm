import jax.numpy as jnp
import requests
from flax import nnx
from PIL import Image
from transformers import SiglipProcessor, SiglipVisionModel

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

    model.eval()
    image_array = jnp.transpose(inputs["pixel_values"].detach().cpu().numpy(), axes=(0, 2, 3, 1))

    image_features_jimm = nnx.jit(model.encode_image)(image_array)

    print(f"Max absolute difference: {jnp.abs(image_features_jimm - image_features_ref).max()}")
    assert jnp.allclose(image_features_jimm, image_features_ref, atol=1e-2), f"Outputs don't match: {image_features_jimm} vs {image_features_ref}"
