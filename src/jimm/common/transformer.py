from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax.typing import DTypeLike
from jaxtyping import Array, Float

from jimm.common.utils import sharded_init


def quickgelu(x: Float[Array, " batch "]) -> Float[Array, " batch "]:
    """Returns the QuickGELU as defined by the OpenAI CLIP model.
    Defined as x * sigmoid(1.702x)

    Returns:
        Float[Array, " batch "]: The output of the quickgelu functions.
    """
    return x * jax.nn.sigmoid(1.702 * x)


class TransformerEncoder(nnx.Module):
    """A Transformer encoder block.

    This implements a standard Transformer encoder.
    """

    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        layernorm_epsilon=1e-5,
        attn_mask: Optional[Float[Array, "seq seq"]] = None,
        use_quick_gelu: bool = False,
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(0),
        mesh: Optional[Mesh] = None,
    ) -> None:
        """Initialize a TransformerEncoder.

        Args:
            hidden_size (int): Size of the hidden dimension.
            mlp_dim (int): Size of the MLP dimension.
            num_heads (int): Number of attention heads.
            dropout_rate (float): Dropout rate. Defaults to 0.0.
            attn_mask (Optional[Float[Array, "seq seq"]]): Optional attention mask.
            use_quick_gelu (bool): Whether to use quickgelu instead of gelu. Defaults to False.
            dtype (DTypeLike): Data type for computations. Defaults to jnp.float32.
            param_dtype (DTypeLike): Data type for parameters. Defaults to jnp.float32.
            rngs (nnx.Rngs): Random number generator keys. Defaults to nnx.Rngs(0).
            mesh (Optional[Mesh]): JAX device mesh for parameter sharding. Defaults to None.
        """
        self.attn_mask = attn_mask
        self.norm1 = nnx.LayerNorm(
            hidden_size,
            epsilon=layernorm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            scale_init=sharded_init(nnx.initializers.ones_init(), P("model"), mesh),
            bias_init=sharded_init(nnx.initializers.zeros_init(), P("model"), mesh),
        )
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_size,
            dropout_rate=dropout_rate,
            broadcast_dropout=False,
            decode=False,
            deterministic=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            kernel_init=sharded_init(nnx.initializers.xavier_uniform(), P(None, "model"), mesh),
            bias_init=sharded_init(nnx.initializers.zeros_init(), P("model"), mesh),
        )
        self.norm2 = nnx.LayerNorm(
            hidden_size,
            epsilon=layernorm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            scale_init=sharded_init(nnx.initializers.ones_init(), P("model"), mesh),
            bias_init=sharded_init(nnx.initializers.zeros_init(), P("model"), mesh),
        )

        activation_fn = quickgelu if use_quick_gelu else nnx.gelu

        self.mlp = nnx.Sequential(
            nnx.Linear(
                hidden_size,
                mlp_dim,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
                kernel_init=sharded_init(nnx.initializers.xavier_uniform(), P(None, "model"), mesh),
                bias_init=sharded_init(nnx.initializers.zeros_init(), P("model"), mesh),
            ),
            activation_fn,
            nnx.Dropout(dropout_rate, rngs=rngs),
            nnx.Linear(
                mlp_dim,
                hidden_size,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
                kernel_init=sharded_init(nnx.initializers.xavier_uniform(), P(None, "model"), mesh),
                bias_init=sharded_init(nnx.initializers.zeros_init(), P("model"), mesh),
            ),
            nnx.Dropout(dropout_rate, rngs=rngs),
        )

    def __call__(self, x: Float[Array, "batch seq hidden"]) -> Float[Array, "batch seq hidden"]:
        """Apply the transformer encoder to the input.

        Args:
            x (Float[Array, "batch seq hidden"]): Input tensor with shape [batch, sequence_length, hidden_size].

        Returns:
            Float[Array, "batch seq hidden"]: Output tensor with the same shape as input.
        """
        seq_len = x.shape[1]
        mask = None
        if self.attn_mask is not None:
            mask_seq_len = min(seq_len, self.attn_mask.shape[0])
            mask = self.attn_mask[:mask_seq_len, :mask_seq_len]
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nnx.Module):
    def __init__(
        self,
        width: int,
        mlp_dim: int,
        layers: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        attn_mask: Optional[Float[Array, "seq seq"]] = None,
        use_quick_gelu: bool = False,
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(0),
        mesh: Optional[Mesh] = None,
    ):
        """Initialize a Transformer.

        Args:
            width (int): The width of the transformer.
            mlp_dim (int): The dimension of the MLP layer.
            layers (int): The number of transformer layers.
            num_heads (int): The number of attention heads.
            dropout_rate (float): The dropout rate. Defaults to 0.0.
            attn_mask (Optional[Float[Array, "seq seq"]]): Optional attention mask.
            use_quick_gelu (bool): Whether to use quickgelu instead of gelu. Defaults to False.
            dtype (DTypeLike): The data type for computations. Defaults to jnp.float32.
            param_dtype (DTypeLike): The data type for parameters. Defaults to jnp.float32.
            rngs (nnx.Rngs): Random number generator keys. Defaults to nnx.Rngs(0).
            mesh (Optional[Mesh]): JAX device mesh for parameter sharding. Defaults to None.
        """
        self.width = width
        self.layers = layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.blocks = nnx.Sequential(
            *[
                TransformerEncoder(
                    hidden_size=width,
                    mlp_dim=mlp_dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    attn_mask=attn_mask,
                    use_quick_gelu=use_quick_gelu,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    rngs=rngs,
                    mesh=mesh,
                )
                for _ in range(layers)
            ]
        )

    def __call__(self, x: Float[Array, "batch seq hidden"]) -> Float[Array, "batch seq hidden"]:
        """Forward pass of the transformer blocks.

        Returns:
            Float[Array, "batch seq hidden"]: The output of the transformer blocks with the same shape as the input.
        """
        return self.blocks(x)
