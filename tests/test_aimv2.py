import tempfile

import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from huggingface_hub import hf_hub_download
from jaxtyping import Array, Float
from safetensors.flax import load_file as load_safetensors
from transformers import AutoModel

from jimm import AIMv2Model

HF_MODEL_NAME = "apple/aimv2-large-patch14-224"
_NATIVE_IMG_SIZE = 224
_LARGE_N_PATCHES = (224 // 14) ** 2

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


@nnx.jit
def _forward(model: AIMv2Model, x: Float[Array, "batch height width channels"]) -> Float[Array, "batch n_patches hidden_size"]:
    return model(x)


def test_aimv2_from_config() -> None:
    """Test AIMv2Model.from_config produces correct output shape.

    Returns:
        None
    """
    model = AIMv2Model.from_config(_CONFIG_SMALL, rngs=nnx.Rngs(0))
    model.eval()
    x = jnp.ones((1, _CONFIG_SMALL["image_size"], _CONFIG_SMALL["image_size"], 3))
    out = model(x)
    assert out.shape == (1, _SMALL_N_PATCHES, _CONFIG_SMALL["hidden_size"])


def test_aimv2_gradient_checkpointing() -> None:
    """Test that use_gradient_checkpointing=True produces numerically identical output to False.

    Returns:
        None
    """
    x = jnp.ones((1, _CONFIG_SMALL["image_size"], _CONFIG_SMALL["image_size"], 3))
    model = AIMv2Model.from_config(_CONFIG_SMALL, rngs=nnx.Rngs(0))
    model_ckpt = AIMv2Model.from_config(_CONFIG_SMALL, use_gradient_checkpointing=True, rngs=nnx.Rngs(1))
    nnx.update(model_ckpt, nnx.state(model))
    model.eval()
    model_ckpt.eval()
    out = model(x)
    out_ckpt = model_ckpt(x)
    assert out.shape == out_ckpt.shape
    assert jnp.allclose(out, out_ckpt, atol=1e-5), f"Checkpointed output differs by up to {jnp.abs(out - out_ckpt).max()}"


def test_aimv2_inference() -> None:
    """Compare AIMv2 patch embeddings with HuggingFace reference at native 224x224.

    Returns:
        None
    """
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
    assert jnp.allclose(jimm_out, jnp.array(hf_out), atol=0.05)


def test_aimv2_save_pretrained_roundtrip() -> None:
    """Test that save_pretrained followed by from_pretrained produces identical outputs.

    Returns:
        None
    """
    model = AIMv2Model.from_pretrained(HF_MODEL_NAME, rngs=nnx.Rngs(0))
    model.eval()

    rng = np.random.default_rng(42)
    x = jnp.array(rng.standard_normal((1, _NATIVE_IMG_SIZE, _NATIVE_IMG_SIZE, 3)).astype(np.float32))

    original_out = _forward(model, x)

    hf_weights = load_safetensors(hf_hub_download(HF_MODEL_NAME, "model.safetensors"))

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)

        saved = load_safetensors(f"{tmpdir}/model.safetensors")
        assert saved["encoder.layers.0.ffn.gate_proj.weight"].shape[1] == 1024
        assert saved["encoder.layers.0.attention.q_proj.weight"].shape == (1024, 1024)
        assert np.allclose(
            saved["encoder.layers.0.attention.q_proj.weight"],
            hf_weights["encoder.layers.0.attention.q_proj.weight"],
            atol=1e-6,
        )

        reloaded = AIMv2Model.from_pretrained(tmpdir, rngs=nnx.Rngs(0))
        reloaded.eval()
        reloaded_out = _forward(reloaded, x)

    assert jnp.allclose(original_out, reloaded_out, atol=1e-5), f"Roundtrip outputs differ by up to {jnp.abs(original_out - reloaded_out).max()}"
