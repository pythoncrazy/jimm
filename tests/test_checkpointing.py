"""Tests for safetensors checkpointing across all models.

This module tests that all models can be saved to and loaded from safetensors
checkpoints, and that they produce identical outputs before and after reloading.
"""

import tempfile
from pathlib import Path

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

# Model identifiers for loading from HuggingFace
VIT_MODEL_NAME = "google/vit-base-patch16-224"
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
SIGLIP_MODEL_NAME = "google/siglip-base-patch16-256"

# Setup mesh for distributed testing
devices = mesh_utils.create_device_mesh((jax.device_count(), 1))
mesh = Mesh(devices, ("data", "fsdp"))
jax.set_mesh(mesh)


def test_vit_checkpoint() -> None:
    """Test VisionTransformer safetensors save/load preserves outputs.

    Saves model to /tmp/, reloads it, and verifies identical outputs.
    """
    print("\n=== Testing VisionTransformer Checkpoint ===")

    # Create original model
    model = VisionTransformer.from_pretrained(VIT_MODEL_NAME, rngs=nnx.Rngs(42))
    model.eval()

    # Load test image
    image = Image.open("images/test_image.jpg")
    processor = ViTImageProcessor.from_pretrained(VIT_MODEL_NAME)
    inputs = processor(images=image, return_tensors="pt")
    x: Float[Array, "batch height width channels"] = jnp.transpose(inputs["pixel_values"].detach().cpu().numpy(), axes=(0, 2, 3, 1))

    # Get output before saving
    output_before = model(x)
    print(f"Output before shape: {output_before.shape}")

    # Save and reload
    with tempfile.TemporaryDirectory(prefix="jimm_vit_", dir="/tmp") as tmpdir:
        save_path = Path(tmpdir) / "vit_checkpoint"
        print(f"Saving to: {save_path}")
        model.save_pretrained(str(save_path))

        # Verify files were created
        assert (save_path / "model.safetensors").exists(), "model.safetensors not found"
        assert (save_path / "config.json").exists(), "config.json not found"
        print("✓ Checkpoint files created")

        # Reload model
        reloaded_model = VisionTransformer.from_pretrained(str(save_path), rngs=nnx.Rngs(42))
        reloaded_model.eval()

        # Get output after reloading
        output_after = reloaded_model(x)
        print(f"Output after shape: {output_after.shape}")

        # Compare outputs
        max_diff = jnp.abs(output_before - output_after).max()
        print(f"Max absolute difference: {max_diff}")

        assert jnp.allclose(output_before, output_after, atol=1e-6), f"VisionTransformer outputs differ after reload: max diff = {max_diff}"

        print("✓ VisionTransformer checkpoint test passed!\n")


def test_clip_vision_checkpoint() -> None:
    """Test CLIPVisionModel safetensors save/load preserves outputs.

    Saves model to /tmp/, reloads it, and verifies identical outputs.
    """
    print("\n=== Testing CLIPVisionModel Checkpoint ===")

    # Create original model
    model = CLIPVisionModel.from_pretrained(CLIP_MODEL_NAME, rngs=nnx.Rngs(42))
    model.eval()

    # Load test image
    image = Image.open("images/test_image.jpg")
    processor = AutoProcessor.from_pretrained(CLIP_MODEL_NAME)
    inputs = processor(images=image, return_tensors="pt")
    x: Float[Array, "batch height width channels"] = jnp.transpose(inputs["pixel_values"].detach().cpu().numpy(), axes=(0, 2, 3, 1))

    # Get output before saving
    output_before = model(x, do_projection=False)
    print(f"Output before shape: {output_before.shape}")

    # Save and reload
    with tempfile.TemporaryDirectory(prefix="jimm_clip_vision_", dir="/tmp") as tmpdir:
        save_path = Path(tmpdir) / "clip_vision_checkpoint"
        print(f"Saving to: {save_path}")
        model.save_pretrained(str(save_path))

        # Verify files were created
        assert (save_path / "model.safetensors").exists(), "model.safetensors not found"
        assert (save_path / "config.json").exists(), "config.json not found"
        print("✓ Checkpoint files created")

        # Reload model
        reloaded_model = CLIPVisionModel.from_pretrained(str(save_path), rngs=nnx.Rngs(42))
        reloaded_model.eval()

        # Get output after reloading
        output_after = reloaded_model(x, do_projection=False)
        print(f"Output after shape: {output_after.shape}")

        # Compare outputs
        max_diff = jnp.abs(output_before - output_after).max()
        print(f"Max absolute difference: {max_diff}")

        assert jnp.allclose(output_before, output_after, atol=1e-6), f"CLIPVisionModel outputs differ after reload: max diff = {max_diff}"

        print("✓ CLIPVisionModel checkpoint test passed!\n")


def test_clip_text_checkpoint() -> None:
    """Test CLIPTextModel safetensors save/load preserves outputs.

    Saves model to /tmp/, reloads it, and verifies identical outputs.
    """
    print("\n=== Testing CLIPTextModel Checkpoint ===")

    # Create original model
    model = CLIPTextModel.from_pretrained(CLIP_MODEL_NAME, rngs=nnx.Rngs(42))
    model.eval()

    # Prepare text inputs
    processor = AutoProcessor.from_pretrained(CLIP_MODEL_NAME)
    text = ["a photo of a cat", "a photo of a dog"]
    inputs = processor(text=text, return_tensors="pt", padding="max_length")
    x: Int[Array, "batch seq_len"] = inputs["input_ids"].detach().cpu().numpy()

    # Get output before saving
    output_before = model(x, do_projection=True)
    print(f"Output before shape: {output_before.shape}")

    # Save and reload
    with tempfile.TemporaryDirectory(prefix="jimm_clip_text_", dir="/tmp") as tmpdir:
        save_path = Path(tmpdir) / "clip_text_checkpoint"
        print(f"Saving to: {save_path}")
        model.save_pretrained(str(save_path))

        # Verify files were created
        assert (save_path / "model.safetensors").exists(), "model.safetensors not found"
        assert (save_path / "config.json").exists(), "config.json not found"
        print("✓ Checkpoint files created")

        # Reload model
        reloaded_model = CLIPTextModel.from_pretrained(str(save_path), rngs=nnx.Rngs(42))
        reloaded_model.eval()

        # Get output after reloading
        output_after = reloaded_model(x, do_projection=True)
        print(f"Output after shape: {output_after.shape}")

        # Compare outputs
        max_diff = jnp.abs(output_before - output_after).max()
        print(f"Max absolute difference: {max_diff}")

        assert jnp.allclose(output_before, output_after, atol=1e-6), f"CLIPTextModel outputs differ after reload: max diff = {max_diff}"

        print("✓ CLIPTextModel checkpoint test passed!\n")


def test_clip_full_checkpoint() -> None:
    """Test full CLIP model safetensors save/load preserves outputs.

    Saves model to /tmp/, reloads it, and verifies identical outputs.
    """
    print("\n=== Testing CLIP (Full Model) Checkpoint ===")

    # Create original model
    model = CLIP.from_pretrained(CLIP_MODEL_NAME, rngs=nnx.Rngs(42))
    model.eval()

    # Prepare inputs
    image = Image.open("images/test_image.jpg")
    processor = AutoProcessor.from_pretrained(CLIP_MODEL_NAME)
    inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt")

    image_array: Float[Array, "batch height width channels"] = jnp.transpose(inputs["pixel_values"].detach().cpu().numpy(), axes=(0, 2, 3, 1))
    text_array: Int[Array, "batch seq_len"] = inputs["input_ids"].detach().cpu().numpy()

    # Get output before saving
    output_before = model(image_array, text_array)
    print(f"Output before shape: {output_before.shape}")

    # Save and reload
    with tempfile.TemporaryDirectory(prefix="jimm_clip_full_", dir="/tmp") as tmpdir:
        save_path = Path(tmpdir) / "clip_full_checkpoint"
        print(f"Saving to: {save_path}")
        model.save_pretrained(str(save_path))

        # Verify files were created
        assert (save_path / "model.safetensors").exists(), "model.safetensors not found"
        assert (save_path / "config.json").exists(), "config.json not found"
        print("✓ Checkpoint files created")

        # Reload model
        reloaded_model = CLIP.from_pretrained(str(save_path), rngs=nnx.Rngs(42))
        reloaded_model.eval()

        # Get output after reloading
        output_after = reloaded_model(image_array, text_array)
        print(f"Output after shape: {output_after.shape}")

        # Compare outputs
        max_diff = jnp.abs(output_before - output_after).max()
        print(f"Max absolute difference: {max_diff}")

        assert jnp.allclose(output_before, output_after, atol=1e-6), f"CLIP (full) outputs differ after reload: max diff = {max_diff}"

        print("✓ CLIP (Full Model) checkpoint test passed!\n")


def test_siglip_vision_checkpoint() -> None:
    """Test SigLIPVisionModel safetensors save/load preserves outputs.

    Saves model to /tmp/, reloads it, and verifies identical outputs.
    """
    print("\n=== Testing SigLIPVisionModel Checkpoint ===")

    # Create original model
    model = SigLIPVisionModel.from_pretrained(SIGLIP_MODEL_NAME, rngs=nnx.Rngs(42))
    model.eval()

    # Load test image
    image = Image.open("images/test_image.jpg")
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)
    inputs = processor(images=image, return_tensors="pt")
    x: Float[Array, "batch height width channels"] = jnp.transpose(inputs["pixel_values"].detach().cpu().numpy(), axes=(0, 2, 3, 1))

    # Get output before saving
    output_before = model(x, do_projection=False)
    print(f"Output before shape: {output_before.shape}")

    # Save and reload
    with tempfile.TemporaryDirectory(prefix="jimm_siglip_vision_", dir="/tmp") as tmpdir:
        save_path = Path(tmpdir) / "siglip_vision_checkpoint"
        print(f"Saving to: {save_path}")
        model.save_pretrained(str(save_path))

        # Verify files were created
        assert (save_path / "model.safetensors").exists(), "model.safetensors not found"
        assert (save_path / "config.json").exists(), "config.json not found"
        print("✓ Checkpoint files created")

        # Reload model
        reloaded_model = SigLIPVisionModel.from_pretrained(str(save_path), rngs=nnx.Rngs(42))
        reloaded_model.eval()

        # Get output after reloading
        output_after = reloaded_model(x, do_projection=False)
        print(f"Output after shape: {output_after.shape}")

        # Compare outputs
        max_diff = jnp.abs(output_before - output_after).max()
        print(f"Max absolute difference: {max_diff}")

        assert jnp.allclose(output_before, output_after, atol=1e-6), f"SigLIPVisionModel outputs differ after reload: max diff = {max_diff}"

        print("✓ SigLIPVisionModel checkpoint test passed!\n")


def test_siglip_text_checkpoint() -> None:
    """Test SigLIPTextModel safetensors save/load preserves outputs.

    Saves model to /tmp/, reloads it, and verifies identical outputs.
    """
    print("\n=== Testing SigLIPTextModel Checkpoint ===")

    # Create original model
    model = SigLIPTextModel.from_pretrained(SIGLIP_MODEL_NAME, rngs=nnx.Rngs(42))
    model.eval()

    # Prepare text inputs
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)
    text = ["a photo of a cat", "a photo of a dog"]
    inputs = processor(text=text, return_tensors="pt", padding="max_length")
    x: Int[Array, "batch seq_len"] = inputs["input_ids"].detach().cpu().numpy()

    # Get output before saving
    output_before = model(x, do_projection=True)
    print(f"Output before shape: {output_before.shape}")

    # Save and reload
    with tempfile.TemporaryDirectory(prefix="jimm_siglip_text_", dir="/tmp") as tmpdir:
        save_path = Path(tmpdir) / "siglip_text_checkpoint"
        print(f"Saving to: {save_path}")
        model.save_pretrained(str(save_path))

        # Verify files were created
        assert (save_path / "model.safetensors").exists(), "model.safetensors not found"
        assert (save_path / "config.json").exists(), "config.json not found"
        print("✓ Checkpoint files created")

        # Reload model
        reloaded_model = SigLIPTextModel.from_pretrained(str(save_path), rngs=nnx.Rngs(42))
        reloaded_model.eval()

        # Get output after reloading
        output_after = reloaded_model(x, do_projection=True)
        print(f"Output after shape: {output_after.shape}")

        # Compare outputs
        max_diff = jnp.abs(output_before - output_after).max()
        print(f"Max absolute difference: {max_diff}")

        assert jnp.allclose(output_before, output_after, atol=1e-6), f"SigLIPTextModel outputs differ after reload: max diff = {max_diff}"

        print("✓ SigLIPTextModel checkpoint test passed!\n")


def test_siglip_full_checkpoint() -> None:
    """Test full SigLIP model safetensors save/load preserves outputs.

    Saves model to /tmp/, reloads it, and verifies identical outputs.
    """
    print("\n=== Testing SigLIP (Full Model) Checkpoint ===")

    # Create original model
    model = SigLIP.from_pretrained(SIGLIP_MODEL_NAME, rngs=nnx.Rngs(42))
    model.eval()

    # Prepare inputs
    image = Image.open("images/test_image.jpg")
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)
    inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt")

    image_array: Float[Array, "batch height width channels"] = jnp.transpose(inputs["pixel_values"].detach().cpu().numpy(), axes=(0, 2, 3, 1))
    text_array: Int[Array, "batch seq_len"] = inputs["input_ids"].detach().cpu().numpy()

    # Get output before saving
    output_before = model(image_array, text_array)
    print(f"Output before shape: {output_before.shape}")

    # Save and reload
    with tempfile.TemporaryDirectory(prefix="jimm_siglip_full_", dir="/tmp") as tmpdir:
        save_path = Path(tmpdir) / "siglip_full_checkpoint"
        print(f"Saving to: {save_path}")
        model.save_pretrained(str(save_path))

        # Verify files were created
        assert (save_path / "model.safetensors").exists(), "model.safetensors not found"
        assert (save_path / "config.json").exists(), "config.json not found"
        print("✓ Checkpoint files created")

        # Reload model
        reloaded_model = SigLIP.from_pretrained(str(save_path), rngs=nnx.Rngs(42))
        reloaded_model.eval()

        # Get output after reloading
        output_after = reloaded_model(image_array, text_array)
        print(f"Output after shape: {output_after.shape}")

        # Compare outputs
        max_diff = jnp.abs(output_before - output_after).max()
        print(f"Max absolute difference: {max_diff}")

        assert jnp.allclose(output_before, output_after, atol=1e-6), f"SigLIP (full) outputs differ after reload: max diff = {max_diff}"

        print("✓ SigLIP (Full Model) checkpoint test passed!\n")


if __name__ == "__main__":
    """Run all checkpoint tests."""
    print("\n" + "=" * 60)
    print("SAFETENSORS CHECKPOINTING TESTS")
    print("=" * 60)

    test_vit_checkpoint()
    test_clip_vision_checkpoint()
    test_clip_text_checkpoint()
    test_clip_full_checkpoint()
    test_siglip_vision_checkpoint()
    test_siglip_text_checkpoint()
    test_siglip_full_checkpoint()

    print("=" * 60)
    print("ALL CHECKPOINT TESTS PASSED! ✓")
    print("=" * 60 + "\n")
