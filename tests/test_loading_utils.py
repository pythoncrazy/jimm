from enum import Enum

import jax.numpy as jnp
import pytest
from flax import nnx

from jimm.common.loading_utils import apply_mapping, expand_scanned_layers


class _Transform(Enum):
    DEFAULT = (None, None, False)


class _SingleParamModel(nnx.Module):
    def __init__(self) -> None:
        self.weight = nnx.Param(jnp.zeros((2, 2), dtype=jnp.float32))


class _ScannedParamModel(nnx.Module):
    def __init__(self) -> None:
        self.layers = nnx.Param(jnp.zeros((2, 3), dtype=jnp.float32))


def test_expand_scanned_layers_expands_scan_batched_blocks() -> None:
    state_dict = {
        "encoder": {
            "layers": {
                "norm1": {
                    "bias": jnp.arange(6, dtype=jnp.float32).reshape(2, 3),
                },
            },
        },
    }

    expanded = expand_scanned_layers(state_dict)

    assert "layers_0" in expanded["encoder"]
    assert "layers_1" in expanded["encoder"]
    assert jnp.array_equal(expanded["encoder"]["layers_0"]["norm1"]["bias"], jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32))
    assert jnp.array_equal(expanded["encoder"]["layers_1"]["norm1"]["bias"], jnp.array([3.0, 4.0, 5.0], dtype=jnp.float32))


def test_expand_scanned_layers_preserves_sequential_layers() -> None:
    state_dict = {
        "encoder": {
            "MAPHead": {
                "mlp": {
                    "layers": {
                        0: {
                            "kernel": jnp.ones((4, 8), dtype=jnp.float32),
                            "bias": jnp.zeros((8,), dtype=jnp.float32),
                        },
                        2: {
                            "kernel": jnp.ones((8, 4), dtype=jnp.float32),
                            "bias": jnp.zeros((4,), dtype=jnp.float32),
                        },
                    },
                },
            },
        },
    }

    expanded = expand_scanned_layers(state_dict)

    mlp_layers = expanded["encoder"]["MAPHead"]["mlp"]["layers"]
    assert 0 in mlp_layers
    assert 2 in mlp_layers
    assert "layers_0" not in expanded["encoder"]["MAPHead"]["mlp"]


def test_apply_mapping_raises_on_incompatible_shape() -> None:
    model = _SingleParamModel()

    with pytest.raises(ValueError, match="Shape mismatch"):
        apply_mapping(
            model,
            {"hf.weight": jnp.ones((3,), dtype=jnp.float32)},
            {"hf\\.weight": ("weight", _Transform.DEFAULT)},
            jnp.float32,
        )


def test_apply_mapping_raises_on_missing_scanned_layers() -> None:
    model = _ScannedParamModel()

    with pytest.raises(ValueError, match="Missing scanned layers"):
        apply_mapping(
            model,
            {"hf.layer0": jnp.ones((3,), dtype=jnp.float32)},
            {"hf\\.layer0": ("layers_0", _Transform.DEFAULT)},
            jnp.float32,
        )
