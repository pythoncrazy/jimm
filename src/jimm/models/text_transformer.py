from typing import Optional

import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from jax.typing import DTypeLike
from jaxtyping import Array, Float

from jimm.common.transformer import TransformerEncoder


class TextTransformer(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        dropout_rate: float = 0.0, # for clip this is zero,
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(0),
        mesh: Optional[Mesh] = None,
    ):
        """Initialize a TextTransformer.

        Args:
            hidden_size (int): Size of the hidden dimension.
            num_layers (int): Number of transformer layers.
            num_heads (int): Number of attention heads.
            dropout_rate (float): Dropout rate. Defaults to 0.0.
            dtype (DTypeLike): Data type for computations. Defaults to jnp.float32.
            param_dtype (DTypeLike): Data type for parameters. Defaults to jnp.float32.
            rngs (nnx.Rngs): Random number generator keys. Defaults to nnx.Rngs(0).
            mesh (Optional[Mesh]): JAX device mesh for parameter sharding. Defaults to None.
        """
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.rngs = rngs
        self.mesh = mesh
        self.resblocks = nnx.Sequential(
            *[
                TransformerEncoder(hidden_size=hidden_size, mlp_dim=hidden_size, num_heads=num_heads, dropout_rate=dropout_rate, dtype=dtype, param_dtype=param_dtype, rngs=rngs, mesh=mesh)
                for _ in range(num_layers)
            ]
        )

    def __call__(self, x: Float[Array, "batch_size seq_len hidden_size"]) -> Float[Array, "batch_size seq_len hidden_size"]:
        """Call the TextTransformer.

        Args:
            x (Float[Array, "batch_size seq_len hidden_size"]): Input tensor.
                The shape is (batch_size, seq_len, hidden_size).

        Returns:
            Float[Array, "batch_size seq_len hidden_size"]: Output tensor.
                The shape is (batch_size, seq_len, hidden_size).
        """
        return self.resblocks(x)
