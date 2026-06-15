import tempfile

import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from huggingface_hub import hf_hub_download
from jaxtyping import Array, Float
from safetensors.flax import load_file as load_safetensors
from transformers import AutoModel

from jimm import DINOv3Model

HF_MODEL_NAME_SMALL = "facebook/dinov3-vits16-pretrain-lvd1689m"
_NATIVE_IMG_SIZE = 224

_CONFIG_SMALL = {
    "hidden_size": 384,
    "num_hidden_layers": 12,
    "num_attention_heads": 6,
    "intermediate_size": 1536,
    "image_size": 224,
    "patch_size": 16,
    "num_register_tokens": 4,
    "rope_theta": 100.0,
    "layerscale_value": 1.0,
    "hidden_act": "gelu",
    "use_gated_mlp": False,
}

_CONFIG_SMALL_GATED = {**_CONFIG_SMALL, "use_gated_mlp": True, "hidden_act": "silu"}


@nnx.jit
def _forward(model: DINOv3Model, x: Float[Array, "batch height width channels"]) -> Float[Array, "batch hidden_size"]:
    return model(x)


def _run_inference_test(hf_model_name: str, atol: float = 0.05) -> None:
    """Load model and compare CLS-token output with HuggingFace reference at native 224x224.

    Args:
        hf_model_name (str): HuggingFace model ID (e.g. "facebook/dinov3-vits16-pretrain-lvd1689m").
        atol (float, optional): Absolute tolerance for numerical comparison. Defaults to 0.05.
    """
    model = DINOv3Model.from_pretrained(hf_model_name, rngs=nnx.Rngs(0))
    model.eval()

    hf_model = AutoModel.from_pretrained(hf_model_name)
    hf_model.eval()

    rng = np.random.default_rng(0)
    img_np = rng.standard_normal((1, 3, _NATIVE_IMG_SIZE, _NATIVE_IMG_SIZE)).astype(np.float32) * 0.5

    with torch.no_grad():
        hf_out = hf_model(pixel_values=torch.from_numpy(img_np)).pooler_output.numpy()

    x: Float[Array, "batch height width channels"] = jnp.transpose(jnp.array(img_np), axes=(0, 2, 3, 1))
    jimm_out = _forward(model, x)

    max_diff = jnp.abs(jimm_out - hf_out).max()
    print(f"[{hf_model_name}] Max absolute difference: {max_diff}")
    assert jnp.allclose(jimm_out, jnp.array(hf_out), atol=atol)


def test_dinov3_small_from_config() -> None:
    """Test DINOv3Model.from_config produces correct output shape for dinov3-small.

    Returns:
        None
    """
    model = DINOv3Model.from_config(_CONFIG_SMALL, rngs=nnx.Rngs(0))
    model.eval()
    x = jnp.ones((1, _CONFIG_SMALL["image_size"], _CONFIG_SMALL["image_size"], 3))
    out = model(x)
    assert out.shape == (1, _CONFIG_SMALL["hidden_size"])


def test_dinov3_gated_mlp_from_config() -> None:
    """Test DINOv3Model.from_config produces correct output shape with gated MLP (SwiGLU).

    Returns:
        None
    """
    model = DINOv3Model.from_config(_CONFIG_SMALL_GATED, rngs=nnx.Rngs(0))
    model.eval()
    x = jnp.ones((1, _CONFIG_SMALL_GATED["image_size"], _CONFIG_SMALL_GATED["image_size"], 3))
    out = model(x)
    assert out.shape == (1, _CONFIG_SMALL_GATED["hidden_size"])


def test_dinov3_variable_image_size() -> None:
    """Test that DINOv3Model produces correct output shapes for multiple image sizes.

    Returns:
        None
    """
    model = DINOv3Model.from_config(_CONFIG_SMALL, rngs=nnx.Rngs(0))
    model.eval()
    for h, w in [(192, 256), (320, 320), (224, 224)]:
        out = model(jnp.ones((1, h, w, 3)))
        assert out.shape == (1, _CONFIG_SMALL["hidden_size"])


def test_dinov3_small_inference() -> None:
    """Compare dinov3-small CLS-token output with HuggingFace reference at native 224x224.

    Returns:
        None
    """
    _run_inference_test(HF_MODEL_NAME_SMALL)


def test_dinov3_small_save_pretrained_roundtrip() -> None:
    """Test that save_pretrained followed by from_pretrained produces identical outputs.

    Returns:
        None
    """
    model = DINOv3Model.from_pretrained(HF_MODEL_NAME_SMALL, rngs=nnx.Rngs(0))
    model.eval()

    rng = np.random.default_rng(42)
    x = jnp.array(rng.standard_normal((1, _NATIVE_IMG_SIZE, _NATIVE_IMG_SIZE, 3)).astype(np.float32))

    original_out = _forward(model, x)

    hf_weights = load_safetensors(hf_hub_download(HF_MODEL_NAME_SMALL, "model.safetensors"))

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)

        saved = load_safetensors(f"{tmpdir}/model.safetensors")
        assert saved["layer.0.mlp.up_proj.weight"].shape[1] == 384
        assert saved["layer.0.attention.q_proj.weight"].shape == (384, 384)
        assert np.allclose(
            saved["layer.0.attention.q_proj.weight"],
            hf_weights["layer.0.attention.q_proj.weight"],
            atol=1e-6,
        )

        reloaded = DINOv3Model.from_pretrained(tmpdir, rngs=nnx.Rngs(0))
        reloaded.eval()
        reloaded_out = _forward(reloaded, x)

    assert jnp.allclose(original_out, reloaded_out, atol=1e-5), f"Roundtrip outputs differ by up to {jnp.abs(original_out - reloaded_out).max()}"
