import jax.numpy as jnp

from jimm.common.loading_utils import expand_scanned_layers


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
