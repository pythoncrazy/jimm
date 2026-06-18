import tempfile

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx
from huggingface_hub import hf_hub_download
from jaxtyping import Array, Float
from safetensors.flax import load_file as load_safetensors
from transformers import AutoModel

from jimm import AIMv2Model

HF_MODEL_NAME = "apple/aimv2-large-patch14-224"
HF_LIT_MODEL_NAME = "apple/aimv2-large-patch14-224-lit"
_NATIVE_IMG_SIZE = 224
_LARGE_N_PATCHES = (224 // 14) ** 2
_LARGE_HIDDEN_SIZE = 1024

_CONFIG_SMALL = {
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "intermediate_size": 128,
    "image_size": 56,
    "patch_size": 14,
    "num_channels": 3,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-5,
}
_SMALL_N_PATCHES = (56 // 14) ** 2

_CONFIG_SMALL_LIT = {
    "model_type": "aimv2",
    "vision_config": _CONFIG_SMALL,
}


@nnx.jit
def _forward(model: AIMv2Model, x: Float[Array, "batch height width channels"]) -> Float[Array, "batch n_patches hidden_size"]:
    return model(x)


@pytest.mark.parametrize("config", [_CONFIG_SMALL, _CONFIG_SMALL_LIT], ids=["vision", "lit"])
def test_aimv2_from_config(config: dict) -> None:
    """Test AIMv2Model.from_config produces correct output shape."""
    model = AIMv2Model.from_config(config, rngs=nnx.Rngs(0))
    model.eval()
    x = jnp.ones((1, _CONFIG_SMALL["image_size"], _CONFIG_SMALL["image_size"], 3))
    out = model(x)
    assert out.shape == (1, _SMALL_N_PATCHES, _CONFIG_SMALL["hidden_size"])


@pytest.mark.parametrize("config", [_CONFIG_SMALL, _CONFIG_SMALL_LIT], ids=["vision", "lit"])
def test_aimv2_gradient_checkpointing(config: dict) -> None:
    """Test that use_gradient_checkpointing=True produces numerically identical output to False."""
    x = jnp.ones((1, _CONFIG_SMALL["image_size"], _CONFIG_SMALL["image_size"], 3))
    model = AIMv2Model.from_config(config, rngs=nnx.Rngs(0))
    model_ckpt = AIMv2Model.from_config(config, use_gradient_checkpointing=True, rngs=nnx.Rngs(1))
    nnx.update(model_ckpt, nnx.state(model))
    model.eval()
    model_ckpt.eval()
    out = model(x)
    out_ckpt = model_ckpt(x)
    assert out.shape == out_ckpt.shape
    assert jnp.allclose(out, out_ckpt, atol=1e-5), f"Checkpointed output differs by up to {jnp.abs(out - out_ckpt).max()}"


@pytest.mark.slow
def test_aimv2_inference() -> None:
    """Compare AIMv2 patch embeddings with HuggingFace reference at native 224x224."""
    model = AIMv2Model.from_pretrained(HF_MODEL_NAME, rngs=nnx.Rngs(0))
    model.eval()

    hf_model = AutoModel.from_pretrained(HF_MODEL_NAME)
    hf_model.eval()

    rng = np.random.default_rng(0)
    img_np = rng.standard_normal((1, 3, _NATIVE_IMG_SIZE, _NATIVE_IMG_SIZE)).astype(np.float32) * 0.5

    with torch.no_grad():
        hf_out = hf_model(pixel_values=torch.from_numpy(img_np)).last_hidden_state.numpy()

    x: Float[Array, "batch height width channels"] = jnp.transpose(jnp.array(img_np), axes=(0, 2, 3, 1))
    jimm_out = _forward(model, x)

    max_diff = jnp.abs(jimm_out - hf_out).max()
    print(f"[{HF_MODEL_NAME}] Max absolute difference: {max_diff}")
    assert jnp.allclose(jimm_out, jnp.array(hf_out), atol=5e-3)


@pytest.mark.slow
def test_aimv2_save_pretrained_roundtrip() -> None:
    """Test that save_pretrained followed by from_pretrained produces identical outputs."""
    model = AIMv2Model.from_pretrained(HF_MODEL_NAME, rngs=nnx.Rngs(0))
    model.eval()

    rng = np.random.default_rng(42)
    x = jnp.array(rng.standard_normal((1, _NATIVE_IMG_SIZE, _NATIVE_IMG_SIZE, 3)).astype(np.float32))

    original_out = _forward(model, x)

    hf_weights = load_safetensors(hf_hub_download(HF_MODEL_NAME, "model.safetensors"))

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)

        saved = load_safetensors(f"{tmpdir}/model.safetensors")
        assert saved["encoder.layers.0.ffn.gate_proj.weight"].shape[1] == _LARGE_HIDDEN_SIZE
        assert saved["encoder.layers.0.attention.q_proj.weight"].shape == (_LARGE_HIDDEN_SIZE, _LARGE_HIDDEN_SIZE)
        assert saved["embeddings.patch_embed.weight"].shape == (_LARGE_HIDDEN_SIZE, 3, 14, 14)
        assert saved["embeddings.rms_norm.weight"].shape == (_LARGE_HIDDEN_SIZE,)
        assert saved["rms_norm.weight"].shape == (_LARGE_HIDDEN_SIZE,)
        for key in ("encoder.layers.0.attention.q_proj.weight", "embeddings.patch_embed.weight", "embeddings.rms_norm.weight", "rms_norm.weight"):
            assert np.allclose(saved[key], hf_weights[key], atol=1e-6), f"{key} differs after save"

        reloaded = AIMv2Model.from_pretrained(tmpdir, rngs=nnx.Rngs(0))
        reloaded.eval()
        reloaded_out = _forward(reloaded, x)

    assert jnp.allclose(original_out, reloaded_out, atol=1e-5), f"Roundtrip outputs differ by up to {jnp.abs(original_out - reloaded_out).max()}"


@pytest.mark.slow
def test_aimv2_lit_inference() -> None:
    """Compare AIMv2 patch embeddings loaded from lit checkpoint against HuggingFace vision encoder.

    Builds the HF reference by loading lit backbone weights into a standard Aimv2VisionModel
    (same architecture, no trust_remote_code required).
    """
    model = AIMv2Model.from_pretrained(HF_LIT_MODEL_NAME, rngs=nnx.Rngs(0))
    model.eval()

    lit_weights = load_safetensors(hf_hub_download(HF_LIT_MODEL_NAME, "model.safetensors"))
    vision_state = {k[len("vision_model.") :]: torch.tensor(np.array(v)) for k, v in lit_weights.items() if k.startswith("vision_model.") and "head" not in k}
    hf_model = AutoModel.from_pretrained(HF_MODEL_NAME)
    hf_model.load_state_dict(vision_state, strict=False)
    hf_model.eval()

    rng = np.random.default_rng(0)
    img_np = rng.standard_normal((1, 3, _NATIVE_IMG_SIZE, _NATIVE_IMG_SIZE)).astype(np.float32) * 0.5

    with torch.no_grad():
        hf_out = hf_model(pixel_values=torch.from_numpy(img_np)).last_hidden_state.numpy()

    x: Float[Array, "batch height width channels"] = jnp.transpose(jnp.array(img_np), axes=(0, 2, 3, 1))
    jimm_out = _forward(model, x)

    max_diff = jnp.abs(jimm_out - hf_out).max()
    print(f"[{HF_LIT_MODEL_NAME}] Max absolute difference: {max_diff}")
    assert jnp.allclose(jimm_out, jnp.array(hf_out), atol=5e-3)


@pytest.mark.slow
def test_aimv2_lit_save_pretrained_roundtrip() -> None:
    """Test that save_pretrained/from_pretrained roundtrip is lossless when loaded from lit checkpoint."""
    model = AIMv2Model.from_pretrained(HF_LIT_MODEL_NAME, rngs=nnx.Rngs(0))
    model.eval()

    rng = np.random.default_rng(42)
    x = jnp.array(rng.standard_normal((1, _NATIVE_IMG_SIZE, _NATIVE_IMG_SIZE, 3)).astype(np.float32))

    original_out = _forward(model, x)

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)

        saved = load_safetensors(f"{tmpdir}/model.safetensors")
        assert saved["encoder.layers.0.ffn.gate_proj.weight"].shape[1] == _LARGE_HIDDEN_SIZE
        assert saved["encoder.layers.0.attention.q_proj.weight"].shape == (_LARGE_HIDDEN_SIZE, _LARGE_HIDDEN_SIZE)

        reloaded = AIMv2Model.from_pretrained(tmpdir, rngs=nnx.Rngs(0))
        reloaded.eval()
        reloaded_out = _forward(reloaded, x)

    assert jnp.allclose(original_out, reloaded_out, atol=1e-5), f"Roundtrip outputs differ by up to {jnp.abs(original_out - reloaded_out).max()}"
