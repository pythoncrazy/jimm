import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from huggingface_hub import hf_hub_download
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jaxtyping import Array, Float
from safetensors.flax import load_file as load_safetensors
from transformers import AutoConfig
from transformers import Dinov2Model as HFDinov2Model

from jimm import DINOv2Model

HF_MODEL_NAME_SMALL = "facebook/dinov2-small"
HF_MODEL_NAME_BASE = "facebook/dinov2-base"
_NATIVE_IMG_SIZE = 518

devices = mesh_utils.create_device_mesh((jax.device_count(), 1))
mesh = Mesh(devices, ("data", "fsdp"))


@nnx.jit
def _forward(model: DINOv2Model, x: Float[Array, "batch height width channels"]) -> Float[Array, "batch hidden_size"]:
    return model(x)


def _run_inference_test(hf_model_name: str, atol: float = 0.05) -> None:
    """Load model and compare CLS-token output with HuggingFace Dinov2Model at native 518x518.

    Args:
        hf_model_name (str): HuggingFace model ID (e.g. "facebook/dinov2-small").
        atol (float, optional): Absolute tolerance for numerical comparison. Defaults to 0.05.
    """
    with mesh:
        model = DINOv2Model.from_pretrained(hf_model_name, rngs=nnx.Rngs(0))
    model.eval()

    hf_model = HFDinov2Model.from_pretrained(hf_model_name)
    hf_model.eval()

    rng = np.random.default_rng(0)
    img_np = rng.standard_normal((1, 3, _NATIVE_IMG_SIZE, _NATIVE_IMG_SIZE)).astype(np.float32) * 0.5

    with torch.no_grad():
        hf_out = hf_model(pixel_values=torch.from_numpy(img_np)).last_hidden_state[:, 0].numpy()

    x: Float[Array, "batch height width channels"] = jnp.transpose(jnp.array(img_np), axes=(0, 2, 3, 1))
    jimm_out = _forward(model, x)

    max_diff = jnp.abs(jimm_out - hf_out).max()
    print(f"[{hf_model_name}] Max absolute difference: {max_diff}")
    assert jnp.allclose(jimm_out, jnp.array(hf_out), atol=atol)


def _run_from_config_test(hf_model_name: str) -> None:
    """Test DINOv2Model.from_config produces correct output shape.

    Args:
        hf_model_name (str): HuggingFace model ID (e.g. "facebook/dinov2-small").
    """
    config = AutoConfig.from_pretrained(hf_model_name).to_dict()
    with mesh:
        model = DINOv2Model.from_config(config, rngs=nnx.Rngs(0))
    model.eval()
    x = jnp.ones((1, config["image_size"], config["image_size"], 3))
    out = model(x)
    assert out.shape == (1, config["hidden_size"])


def test_dinov2_small_inference() -> None:
    """Compare dinov2-small CLS-token output with HuggingFace reference at native 518x518.

    Returns:
        None
    """
    _run_inference_test(HF_MODEL_NAME_SMALL)


def test_dinov2_small_from_config() -> None:
    """Test DINOv2Model.from_config produces correct output shape for dinov2-small.

    Returns:
        None
    """
    _run_from_config_test(HF_MODEL_NAME_SMALL)


def test_dinov2_base_inference() -> None:
    """Compare dinov2-base CLS-token output with HuggingFace reference at native 518x518.

    dinov2-base uses hidden_size=768, num_heads=12, mlp_dim=3072 vs small's 384/6/1536.
    Larger matrix dimensions accumulate slightly more float32 error, so atol=0.06 is used.

    Returns:
        None
    """
    _run_inference_test(HF_MODEL_NAME_BASE, atol=0.06)


def test_dinov2_base_from_config() -> None:
    """Test DINOv2Model.from_config produces correct output shape for dinov2-base.

    Returns:
        None
    """
    _run_from_config_test(HF_MODEL_NAME_BASE)


def test_dinov2_small_save_pretrained_roundtrip() -> None:
    """Test that save_pretrained followed by from_pretrained produces identical outputs.

    Returns:
        None
    """
    with mesh:
        model = DINOv2Model.from_pretrained(HF_MODEL_NAME_SMALL, rngs=nnx.Rngs(0))
    model.eval()

    rng = np.random.default_rng(42)
    x = jnp.array(rng.standard_normal((1, _NATIVE_IMG_SIZE, _NATIVE_IMG_SIZE, 3)).astype(np.float32))

    original_out = _forward(model, x)

    hf_weights = load_safetensors(hf_hub_download(HF_MODEL_NAME_SMALL, "model.safetensors"))

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)

        saved = load_safetensors(f"{tmpdir}/model.safetensors")
        # Verify HF-format shapes: Linear weights are (out, in), QKV is (hidden, hidden)
        assert saved["encoder.layer.0.mlp.fc1.weight"].shape == (1536, 384)
        assert saved["encoder.layer.0.attention.attention.query.weight"].shape == (384, 384)
        # Verify values match the HF originals exactly
        assert np.allclose(saved["encoder.layer.0.mlp.fc1.weight"], hf_weights["encoder.layer.0.mlp.fc1.weight"], atol=1e-6)
        assert np.allclose(saved["encoder.layer.0.attention.attention.query.weight"], hf_weights["encoder.layer.0.attention.attention.query.weight"], atol=1e-6)

        with mesh:
            reloaded = DINOv2Model.from_pretrained(tmpdir, rngs=nnx.Rngs(0))
        reloaded.eval()
        reloaded_out = _forward(reloaded, x)

    assert jnp.allclose(original_out, reloaded_out, atol=1e-5), f"Roundtrip outputs differ by up to {jnp.abs(original_out - reloaded_out).max()}"
