import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib
from jax.typing import DTypeLike
from jaxtyping import Array, Float

from jimm.common.sharding import NoSharding, ShardingSpec, reshard_like


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
        attention_fn: Callable[..., Any] | None = None,
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
            use_gradient_checkpointing (bool, optional): Whether to checkpoint the attention and MLP sublayers. Defaults to False.
            attention_fn (Callable[..., Any] | None, optional): Custom attention function compatible with
                nnx.MultiHeadAttention's attention_fn interface (e.g. jimm.tokamax_attention or jimm.make_tokamax_attention("mosaic_tpu")).
                When provided, the causal flag is set automatically based on whether attn_mask is not None,
                and the flax mask is not passed to the attention layer. Defaults to None (uses nnx.dot_product_attention).
            rngs (rnglib.Rngs | None, optional): Random number generator keys. If None, initializes to nnx.Rngs(0).
            dtype (DTypeLike, optional): Data type for computations. Defaults to jnp.float32.
            param_dtype (DTypeLike, optional): Data type for parameters. Defaults to jnp.float32.
            sharding (ShardingSpec, optional): Sharding specification for parameters. Defaults to NoSharding.
        """
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.attn_mask = nnx.Variable(attn_mask) if attn_mask is not None else None
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self._skip_flax_mask = attention_fn is not None
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
            attention_fn=functools.partial(attention_fn, causal=attn_mask is not None) if attention_fn is not None else nnx.dot_product_attention,
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
        """Apply the transformer encoder to the input.

        Args:
            x (Float[Array, "batch seq hidden"]): Input tensor with shape [batch, sequence_length, hidden_size].

        Returns:
            Float[Array, "batch seq hidden"]: Output tensor with the same shape as input.
        """
        seq_len = x.shape[1]
        mask = None
        if self.attn_mask is not None and not self._skip_flax_mask:
            attn_mask_val = self.attn_mask[...]
            mask_seq_len = min(seq_len, attn_mask_val.shape[0])
            mask = attn_mask_val[:mask_seq_len, :mask_seq_len]

        if self.use_gradient_checkpointing:
            attn_out = jax.checkpoint(lambda hidden: self.attn(self.norm1(hidden), mask=mask))(x)
            attn_out = reshard_like(attn_out, x)
            x = x + attn_out
            mlp_out = jax.checkpoint(lambda hidden: self.mlp(self.norm2(hidden)))(x)
            mlp_out = reshard_like(mlp_out, x)
            return x + mlp_out

        attn_out = reshard_like(self.attn(self.norm1(x), mask=mask), x)
        x = x + attn_out
        mlp_out = reshard_like(self.mlp(self.norm2(x)), x)
        x = x + mlp_out
        return x


@nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
def scan_forward(
    x: Float[Array, "batch seq hidden"],
    layer: TransformerEncoder,
) -> Float[Array, "batch seq hidden"]:
    """Apply a single TransformerEncoder layer inside an nnx.scan loop.

    Args:
        x (Float[Array, "batch seq hidden"]): Carry tensor passed through all layers.
        layer (TransformerEncoder): Batched layer module scanned over axis 0.

    Returns:
        Float[Array, "batch seq hidden"]: Updated carry after applying ``layer``.
    """
    return layer(x)


@nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
def scan_forward_remat(
    x: Float[Array, "batch seq hidden"],
    layer: TransformerEncoder,
) -> Float[Array, "batch seq hidden"]:
    """Apply a single TransformerEncoder layer with gradient checkpointing inside an nnx.scan loop.

    Args:
        x (Float[Array, "batch seq hidden"]): Carry tensor passed through all layers.
        layer (TransformerEncoder): Batched layer module scanned over axis 0.

    Returns:
        Float[Array, "batch seq hidden"]: Updated carry after applying ``layer`` with rematerialization.
    """
    return jax.checkpoint(layer)(x)


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
        attention_fn: Callable[..., Any] | None = None,
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
            attention_fn (Callable[..., Any] | None, optional): Custom attention function. Defaults to None.
            rngs (rnglib.Rngs | None, optional): Random number generator keys. If None, initializes to nnx.Rngs(0).
            dtype (DTypeLike, optional): The data type for computations. Defaults to jnp.float32.
            param_dtype (DTypeLike, optional): The data type for parameters. Defaults to jnp.float32.
            sharding (ShardingSpec, optional): Sharding specification for parameters. Defaults to NoSharding.
        """
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing

        @nnx.split_rngs(splits=num_layers)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_block(rngs: rnglib.Rngs) -> TransformerEncoder:
            return TransformerEncoder(
                hidden_size=hidden_size,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                layernorm_epsilon=layernorm_epsilon,
                dropout_rate=dropout_rate,
                attn_mask=attn_mask,
                use_quick_gelu=use_quick_gelu,
                use_gradient_checkpointing=False,
                attention_fn=attention_fn,
                rngs=rngs,
                dtype=dtype,
                param_dtype=param_dtype,
                sharding=sharding,
            )

        self.layers = create_block(rngs)
        for _, var in nnx.iter_graph(self.layers):
            if isinstance(var, nnx.Variable) and var.get_metadata().get("out_sharding") is not None:
                var.set_metadata(out_sharding=(None,) + tuple(var.get_metadata()["out_sharding"]))

    def __call__(self, x: Float[Array, "batch seq hidden"]) -> Float[Array, "batch seq hidden"]:
        """Forward pass applying all transformer blocks via scan.

        Args:
            x (Float[Array, "batch seq hidden"]): Input tensor.

        Returns:
            Float[Array, "batch seq hidden"]: The output of the transformer blocks with the same shape as the input.
        """
        if self.use_gradient_checkpointing:
            return scan_forward_remat(x, self.layers)
        return scan_forward(x, self.layers)
