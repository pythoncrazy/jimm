"""Tests for safetensors checkpointing across all models."""

import tempfile
from pathlib import Path
from typing import Callable, Protocol, Self, TypeVar

import jax
import jax.numpy as jnp
from flax import nnx
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jaxtyping import Array, Float, Int
from PIL import Image
from transformers import AutoProcessor, ViTImageProcessor

from jimm.models.clip import CLIP, CLIPTextModel, CLIPVisionModel
from jimm.models.siglip import SigLIP, SigLIPTextModel, SigLIPVisionModel
from jimm.models.vit import VisionTransformer

VIT_MODEL_NAME = "google/vit-base-patch16-224"
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
SIGLIP_MODEL_NAME = "google/siglip-base-patch16-256"

devices = mesh_utils.create_device_mesh((jax.device_count(), 1))
mesh = Mesh(devices, ("data", "fsdp"))
jax.set_mesh(mesh)


class CheckpointableModule(Protocol):
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, rngs: nnx.Rngs | None = None) -> Self: ...

    def eval(self) -> None: ...

    def save_pretrained(self, save_directory: str) -> None: ...


M = TypeVar("M", bound=CheckpointableModule)


def _load_image(model_name: str) -> Float[Array, "batch height width channels"]:
    """Load and preprocess test image for a given model."""
    image = Image.open("images/test_image.jpg")
    if model_name == VIT_MODEL_NAME:
        processor = ViTImageProcessor.from_pretrained(model_name)
    else:
        processor = AutoProcessor.from_pretrained(model_name)
    inputs = processor(images=image, return_tensors="pt")
    return jnp.transpose(inputs["pixel_values"].detach().cpu().numpy(), axes=(0, 2, 3, 1))


def _load_text(model_name: str) -> Int[Array, "batch seq_len"]:
    """Load and tokenize test text for a given model."""
    processor = AutoProcessor.from_pretrained(model_name)
    inputs = processor(text=["a photo of a cat", "a photo of a dog"], return_tensors="pt", padding="max_length")
    return inputs["input_ids"].detach().cpu().numpy()


def _checkpoint_roundtrip(
    model_cls: type[M],
    model_name: str,
    forward_fn: Callable[[M], Array],
    prefix: str,
) -> None:
    """Test that a model produces identical outputs after save/load cycle."""
    model = model_cls.from_pretrained(model_name, rngs=nnx.Rngs(42))
    model.eval()
    output_before = forward_fn(model)

    with tempfile.TemporaryDirectory(prefix=prefix, dir="/tmp") as tmpdir:
        save_path = Path(tmpdir) / "checkpoint"
        model.save_pretrained(str(save_path))

        assert (save_path / "model.safetensors").exists()
        assert (save_path / "config.json").exists()

        reloaded = model_cls.from_pretrained(str(save_path), rngs=nnx.Rngs(42))
        reloaded.eval()
        output_after = forward_fn(reloaded)

        assert jnp.allclose(output_before, output_after, atol=1e-6)


def test_vit_checkpoint() -> None:
    """Test VisionTransformer checkpoint save/load preserves outputs."""
    x = _load_image(VIT_MODEL_NAME)
    _checkpoint_roundtrip(VisionTransformer, VIT_MODEL_NAME, lambda m: m(x), "jimm_vit_")


def test_clip_vision_checkpoint() -> None:
    """Test CLIPVisionModel checkpoint save/load preserves outputs."""
    x = _load_image(CLIP_MODEL_NAME)
    _checkpoint_roundtrip(CLIPVisionModel, CLIP_MODEL_NAME, lambda m: m(x, do_projection=False), "jimm_clip_vision_")


def test_clip_text_checkpoint() -> None:
    """Test CLIPTextModel checkpoint save/load preserves outputs."""
    x = _load_text(CLIP_MODEL_NAME)
    _checkpoint_roundtrip(CLIPTextModel, CLIP_MODEL_NAME, lambda m: m(x, do_projection=True), "jimm_clip_text_")


def test_clip_full_checkpoint() -> None:
    """Test full CLIP model checkpoint save/load preserves outputs."""
    image = _load_image(CLIP_MODEL_NAME)
    text = _load_text(CLIP_MODEL_NAME)
    _checkpoint_roundtrip(CLIP, CLIP_MODEL_NAME, lambda m: m(image, text), "jimm_clip_full_")


def test_siglip_vision_checkpoint() -> None:
    """Test SigLIPVisionModel checkpoint save/load preserves outputs."""
    x = _load_image(SIGLIP_MODEL_NAME)
    _checkpoint_roundtrip(SigLIPVisionModel, SIGLIP_MODEL_NAME, lambda m: m(x, do_projection=False), "jimm_siglip_vision_")


def test_siglip_text_checkpoint() -> None:
    """Test SigLIPTextModel checkpoint save/load preserves outputs."""
    x = _load_text(SIGLIP_MODEL_NAME)
    _checkpoint_roundtrip(SigLIPTextModel, SIGLIP_MODEL_NAME, lambda m: m(x, do_projection=True), "jimm_siglip_text_")


def test_siglip_full_checkpoint() -> None:
    """Test full SigLIP model checkpoint save/load preserves outputs."""
    image = _load_image(SIGLIP_MODEL_NAME)
    text = _load_text(SIGLIP_MODEL_NAME)
    _checkpoint_roundtrip(SigLIP, SIGLIP_MODEL_NAME, lambda m: m(image, text), "jimm_siglip_full_")


if __name__ == "__main__":
    test_vit_checkpoint()
    test_clip_vision_checkpoint()
    test_clip_text_checkpoint()
    test_clip_full_checkpoint()
    test_siglip_vision_checkpoint()
    test_siglip_text_checkpoint()
    test_siglip_full_checkpoint()
