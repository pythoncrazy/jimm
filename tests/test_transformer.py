import jax
import jax.numpy as jnp
from flax import nnx
from jax.experimental import mesh_utils
from jax.sharding import Mesh

from jimm.common.transformer import Transformer, TransformerEncoder

mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(), 1)), ("data", "fsdp"))
jax.set_mesh(mesh)


def test_transformer_encoder_gradient_checkpointing_matches_eager() -> None:
    x = jnp.ones((2, 4, 8), dtype=jnp.float32)

    with mesh:
        eager = TransformerEncoder(
            hidden_size=8,
            mlp_dim=16,
            num_heads=2,
            use_gradient_checkpointing=False,
            rngs=nnx.Rngs(0),
        )
        checkpointed = TransformerEncoder(
            hidden_size=8,
            mlp_dim=16,
            num_heads=2,
            use_gradient_checkpointing=True,
            rngs=nnx.Rngs(1),
        )

    nnx.update(checkpointed, nnx.state(eager, nnx.Param))

    assert jnp.allclose(eager(x), checkpointed(x), atol=1e-6)


def test_transformer_gradient_checkpointing_matches_eager() -> None:
    x = jnp.ones((2, 4, 8), dtype=jnp.float32)

    with mesh:
        eager = Transformer(
            hidden_size=8,
            mlp_dim=16,
            num_layers=2,
            num_heads=2,
            use_gradient_checkpointing=False,
            rngs=nnx.Rngs(0),
        )
        checkpointed = Transformer(
            hidden_size=8,
            mlp_dim=16,
            num_layers=2,
            num_heads=2,
            use_gradient_checkpointing=True,
            rngs=nnx.Rngs(1),
        )

    nnx.update(checkpointed, nnx.state(eager, nnx.Param))

    assert jnp.allclose(eager(x), checkpointed(x), atol=1e-6)
