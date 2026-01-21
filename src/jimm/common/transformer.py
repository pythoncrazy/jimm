import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib
from jax.typing import DTypeLike
from jaxtyping import Array, Float

from jimm.common.sharding import NoSharding, ShardingSpec


def quickgelu(x: Float[Array, " batch "]) -> Float[Array, " batch "]:
    """Returns the QuickGELU as defined by the OpenAI CLIP model.

    Defined as x * sigmoid(1.702x).

    Args:
        x (Float[Array, " batch "]): Input tensor.

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
        layernorm_epsilon: float = 1e-5,
        dropout_rate: float = 0.0,
        attn_mask: Float[Array, "seq seq"] | None = None,
        use_quick_gelu: bool = False,
        use_gradient_checkpointing: bool = False,
        rngs: rnglib.Rngs | None = None,
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        sharding: ShardingSpec = NoSharding,
    ) -> None:
        """Initialize a TransformerEncoder.

        Args:
            hidden_size (int): Size of the hidden dimension.
            mlp_dim (int): Size of the MLP dimension.
            num_heads (int): Number of attention heads.
            layernorm_epsilon (float, optional): The epsilon used in layernorm calculation. Defaults to 1e-5.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.0.
            attn_mask (Float[Array, "seq seq"] | None, optional): Optional attention mask. Defaults to None.
            use_quick_gelu (bool, optional): Whether to use quickgelu instead of gelu. Defaults to False.
            use_gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
            rngs (rnglib.Rngs | None, optional): Random number generator keys. If None, initializes to nnx.Rngs(0).
            dtype (DTypeLike, optional): Data type for computations. Defaults to jnp.float32.
            param_dtype (DTypeLike, optional): Data type for parameters. Defaults to jnp.float32.
            sharding (ShardingSpec, optional): Sharding specification for parameters. Defaults to NoSharding.
        """
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.attn_mask = attn_mask
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.norm1 = nnx.LayerNorm(
            hidden_size,
            epsilon=layernorm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            scale_init=nnx.with_partitioning(
                nnx.initializers.ones_init(),
                sharding.layernorm,
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros_init(),
                sharding.layernorm,
            ),
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
            kernel_init=nnx.with_partitioning(
                nnx.initializers.xavier_uniform(),
                sharding.attn_qkv_kernel,
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros_init(),
                sharding.attn_qkv_bias,
            ),
            out_kernel_init=nnx.with_partitioning(
                nnx.initializers.xavier_uniform(),
                sharding.attn_out_kernel,
            ),
            out_bias_init=nnx.with_partitioning(
                nnx.initializers.zeros_init(),
                sharding.attn_out_bias,
            ),
        )
        self.norm2 = nnx.LayerNorm(
            hidden_size,
            epsilon=layernorm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            scale_init=nnx.with_partitioning(
                nnx.initializers.ones_init(),
                sharding.layernorm,
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros_init(),
                sharding.layernorm,
            ),
        )

        activation_fn = quickgelu if use_quick_gelu else nnx.gelu

        self.mlp = nnx.Sequential(
            nnx.Linear(
                hidden_size,
                mlp_dim,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
                kernel_init=nnx.with_partitioning(
                    nnx.initializers.xavier_uniform(),
                    sharding.mlp_up_kernel,
                ),
                bias_init=nnx.with_partitioning(
                    nnx.initializers.zeros_init(),
                    sharding.mlp_up_bias,
                ),
            ),
            activation_fn,
            nnx.Dropout(dropout_rate, rngs=rngs),
            nnx.Linear(
                mlp_dim,
                hidden_size,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
                kernel_init=nnx.with_partitioning(
                    nnx.initializers.xavier_uniform(),
                    sharding.mlp_down_kernel,
                ),
                bias_init=nnx.with_partitioning(
                    nnx.initializers.zeros_init(),
                    sharding.mlp_down_bias,
                ),
            ),
            nnx.Dropout(dropout_rate, rngs=rngs),
        )

    def __call__(self, x: Float[Array, "batch seq hidden"]) -> Float[Array, "batch seq hidden"]:
        """Apply the transformer encoder to the input with optional gradient checkpointing.

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

        if self.use_gradient_checkpointing:
            attn_out = jax.checkpoint(lambda x: self.attn(self.norm1(x), mask=mask))(x)
            x = x + attn_out
            mlp_out = jax.checkpoint(lambda x: self.mlp(self.norm2(x)))(x)
            x = x + mlp_out
        else:
            x = x + self.attn(self.norm1(x), mask=mask)
            x = x + self.mlp(self.norm2(x))

        return x


class Transformer(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        num_layers: int,
        num_heads: int,
        layernorm_epsilon: float = 1e-6,
        dropout_rate: float = 0.0,
        attn_mask: Float[Array, "seq seq"] | None = None,
        use_quick_gelu: bool = False,
        use_gradient_checkpointing: bool = False,
        rngs: rnglib.Rngs | None = None,
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        sharding: ShardingSpec = NoSharding,
    ):
        """Initialize a Transformer.

        Args:
            hidden_size (int): The hidden dimension size of the transformer.
            mlp_dim (int): The dimension of the MLP layer.
            num_layers (int): The number of transformer layers.
            num_heads (int): The number of attention heads.
            layernorm_epsilon (float, optional): The epsilon used in layernorm calculation. Defaults to 1e-6.
            dropout_rate (float, optional): The dropout rate. Defaults to 0.0.
            attn_mask (Float[Array, "seq seq"] | None, optional): Optional attention mask. Defaults to None.
            use_quick_gelu (bool, optional): Whether to use quickgelu instead of gelu. Defaults to False.
            use_gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
            rngs (rnglib.Rngs | None, optional): Random number generator keys. If None, initializes to nnx.Rngs(0).
            dtype (DTypeLike, optional): The data type for computations. Defaults to jnp.float32.
            param_dtype (DTypeLike, optional): The data type for parameters. Defaults to jnp.float32.
            sharding (ShardingSpec, optional): Sharding specification for parameters. Defaults to NoSharding.
        """
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing

        for i in range(self.num_layers):
            layer = TransformerEncoder(
                hidden_size=hidden_size,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                layernorm_epsilon=layernorm_epsilon,
                dropout_rate=dropout_rate,
                attn_mask=attn_mask,
                use_quick_gelu=use_quick_gelu,
                use_gradient_checkpointing=use_gradient_checkpointing,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
                sharding=sharding,
            )
            setattr(self, f"layers_{i}", layer)

    def __call__(self, x: Float[Array, "batch seq hidden"]) -> Float[Array, "batch seq hidden"]:
        """Forward pass of the transformer blocks with optional gradient checkpointing.

        Args:
            x (Float[Array, "batch seq hidden"]): Input tensor.

        Returns:
            Float[Array, "batch seq hidden"]: The output of the transformer blocks with the same shape as the input.
        """
        if self.use_gradient_checkpointing:
            for i in range(self.num_layers):
                layer = getattr(self, f"layers_{i}")
                x = jax.checkpoint(layer)(x)
            return x
        else:
            for i in range(self.num_layers):
                layer = getattr(self, f"layers_{i}")
                x = layer(x)
            return x
