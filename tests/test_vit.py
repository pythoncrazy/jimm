import jax.numpy as jnp
import pytest
import requests
from flax import nnx
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

from jimm.models.vit import VisionTransformer

HF_MODEL_B16_224 = "google/vit-base-patch16-224"
SAFETENSORS_B16_224 = "weights/model-base-16-224.safetensors"
IMG_SIZE_224 = 224
HF_MODEL_B32_384 = "google/vit-base-patch32-384"
IMG_SIZE_384 = 384


@pytest.mark.parametrize(
    "model_to_load, use_pytorch, hf_processor_model_name, img_size_val, atol",
    [
        (SAFETENSORS_B16_224, False, HF_MODEL_B16_224, IMG_SIZE_224, 0.05),
        (HF_MODEL_B16_224, False, HF_MODEL_B16_224, IMG_SIZE_224, 0.05),
        (HF_MODEL_B32_384, True, HF_MODEL_B32_384, IMG_SIZE_384, 0.05),
    ],
)
def test_vision_transformer_inference(model_to_load, use_pytorch, hf_processor_model_name, img_size_val, atol):
    model = VisionTransformer.from_pretrained(model_to_load, use_pytorch=use_pytorch)

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    processor = ViTImageProcessor.from_pretrained(hf_processor_model_name)

    inputs = processor(
        images=image,
        return_tensors="pt",
        size={"height": img_size_val, "width": img_size_val},
        do_resize=True,
    )

    pytorch_model = ViTForImageClassification.from_pretrained(hf_processor_model_name)
    pytorch_model.eval()
    outputs = pytorch_model(**inputs)
    logits_ref = outputs.logits.detach().cpu().numpy()

    model.eval()
    x_eval = jnp.transpose(inputs["pixel_values"].detach().cpu().numpy(), axes=(0, 2, 3, 1))
    logits_flax = nnx.jit(model)(x_eval)

    max_abs_diff = jnp.abs(logits_flax - logits_ref).max()
    print(f"Testing with model_to_load: {model_to_load}, use_pytorch: {use_pytorch}, hf_processor: {hf_processor_model_name}, img_size: {img_size_val}")
    print(f"Max absolute difference: {max_abs_diff}")
    assert max_abs_diff < atol
