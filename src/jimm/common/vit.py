import functools
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, DTypeLike, Float

from jimm.common.sharding import NoSharding, ShardingSpec, named_sharding_like, reshard_like, sharding_of
from jimm.common.transformer import Transformer, scan_forward, scan_forward_remat


class MultiHeadAttentionPoolingHead(nnx.Module):
    """Multihead Attention Pooling, as needed by the SigLIP model"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        layernorm_epsilon: float = 1e-6,
        attention_fn: Callable[..., Any] | None = None,
        rngs: rnglib.Rngs | None = None,
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        sharding: ShardingSpec = NoSharding,
    ):
        """Initialization of the Multihead Attention Pooling.

        Args:
            hidden_size (int): The size of the hidden layer, which determines the dimensionality of the model's internal representations.
            intermediate_size (int): The dimension of the intermediate MLP at the end of the MAP head.
            num_heads (int): The number of attention heads.
            layernorm_epsilon (float, optional): The epsilon used in the layernorm. Defaults to 1e-6.
            attention_fn (Callable[..., Any] | None, optional): Custom attention function compatible with
                nnx.MultiHeadAttention's attention_fn interface (e.g. jimm.tokamax_attention or jimm.make_tokamax_attention("mosaic_tpu")).
                Defaults to None (uses nnx.dot_product_attention).
            rngs (rnglib.Rngs | None, optional): The flax nnx rng to use for initialization. If None, initializes to nnx.Rngs(0).
            dtype (DTypeLike, optional): The data type for computations. Defaults to jnp.float32.
            param_dtype (DTypeLike, optional): The data type for parameters. Defaults to jnp.float32.
            sharding (ShardingSpec, optional): Sharding specification for parameters. Defaults to NoSharding.
        """
        if rngs is None:
            rngs = nnx.Rngs(0)
        probe_value: Float[Array, "1 1 hidden_size"] = nnx.initializers.zeros_init()(rngs.params(), (1, 1, hidden_size))
        self.probe = nnx.Param(probe_value, out_sharding=sharding.probe_token)

        self.attn = nnx.MultiHeadAttention(
            num_heads,
            hidden_size,
            broadcast_dropout=False,
            decode=False,
            deterministic=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            attention_fn=functools.partial(attention_fn, causal=False) if attention_fn is not None else nnx.dot_product_attention,
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

        self.layernorm = nnx.LayerNorm(
            num_features=hidden_size,
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

        self.mlp = nnx.Sequential(
            nnx.Linear(
                hidden_size,
                intermediate_size,
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
            nnx.gelu,
            nnx.Linear(
                intermediate_size,
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
        )

    def __call__(self, hidden_state: Float[Array, "batch length hidden_size"]) -> Array:
        """Apply the MAP head to produce a pooled representation.

        Args:
            hidden_state (Float[Array, "batch length hidden_size"]): Sequence of patch embeddings.

        Returns:
            Float[Array, "batch hidden_size"]: Pooled output embedding for each item in the batch.
        """
        batch_size = hidden_state.shape[0]
        probe: Float[Array, "batch 1 hidden_size"] = jnp.tile(self.probe[...], [batch_size, 1, 1])
        probe = reshard_like(probe, hidden_state)
        x: Float[Array, "batch 1 hidden_size"] = self.attn(probe, hidden_state, hidden_state, decode=False)
        residual = x
        x: Float[Array, "batch 1 hidden_size"] = self.layernorm(x)
        x = residual + reshard_like(self.mlp(x), residual)
        return x[:, 0]


class VisionTransformerBase(nnx.Module):
    """A base Vision Transformer (ViT) model."""

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        mlp_dim: int,
        layernorm_epsilon: float = 1e-5,
        pooling_type: str = "CLS",
        dropout_rate: float = 0.0,
        use_quick_gelu: bool = False,
        use_pre_norm: bool = False,
        use_patch_bias: bool = True,
        use_gradient_checkpointing: bool = False,
        attention_fn: Callable[..., Any] | None = None,
        rngs: rnglib.Rngs | None = None,
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        sharding: ShardingSpec = NoSharding,
    ):
        """Initialize the Vision Transformer base model.

        Args:
            img_size (int): The size of the input images.
            patch_size (int): The patch size of the vision transformer.
            in_channels (int): The number of input channels.
            hidden_size (int): The width of the vision transformer.
            num_layers (int): The number of layers in the vision transformer.
            num_heads (int): The number of attention heads in the vision transformer.
            mlp_dim (int): The dimension of the MLP in the transformer blocks.
            layernorm_epsilon (float, optional): Epsilon for LayerNorm. Defaults to 1e-5.
            pooling_type (str, optional): The pooling method, either CLS or MAP. Defaults to "CLS".
            dropout_rate (float, optional): The dropout rate. Defaults to 0.0.
            use_quick_gelu (bool, optional): Whether to use QuickGELU activation. Defaults to False.
            use_pre_norm (bool, optional): Whether to apply LayerNorm before the transformer. Defaults to False.
            use_patch_bias (bool, optional): Whether to use bias in the patch embedding convolution. Defaults to True.
            use_gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
            attention_fn (Callable[..., Any] | None, optional): Custom attention function compatible with
                nnx.MultiHeadAttention's attention_fn interface (e.g. jimm.tokamax_attention or jimm.make_tokamax_attention("mosaic_tpu")).
                Defaults to None (uses nnx.dot_product_attention).
            rngs (rnglib.Rngs | None, optional): The random number generator state. If None, initializes to nnx.Rngs(0).
            dtype (DTypeLike, optional): The data type for computations. Defaults to jnp.float32.
            param_dtype (DTypeLike, optional): The data type for parameters. Defaults to jnp.float32.
            sharding (ShardingSpec, optional): Sharding specification for parameters. Defaults to NoSharding.
        """
        if rngs is None:
            rngs = nnx.Rngs(0)
        n_patches: int = (img_size // patch_size) ** 2
        self.use_pre_norm = use_pre_norm
        self.pooling_type = pooling_type

        self.patch_embeddings = nnx.Conv(
            in_features=in_channels,
            out_features=hidden_size,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            use_bias=use_patch_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.xavier_uniform(),
                sharding.patch_conv_kernel,
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros_init(),
                sharding.patch_conv_bias,
            ),
        )
        if self.pooling_type == "CLS":
            cls_token_value: Float[Array, "1 1 hidden_size"] = nnx.initializers.zeros_init()(rngs.params(), (1, 1, hidden_size))
            self.cls_token = nnx.Param(cls_token_value, out_sharding=sharding.cls_token)
            pos_emb_value: Float[Array, "1 n_patches+1 hidden_size"] = nnx.initializers.truncated_normal(stddev=0.02)(rngs.params(), (1, n_patches + 1, hidden_size))
        elif self.pooling_type == "MAP":
            pos_emb_value: Float[Array, "1 n_patches hidden_size"] = nnx.initializers.truncated_normal(stddev=0.02)(rngs.params(), (1, n_patches, hidden_size))
            self.map_head = MultiHeadAttentionPoolingHead(
                hidden_size=hidden_size,
                intermediate_size=4 * hidden_size,
                num_heads=num_heads,
                layernorm_epsilon=layernorm_epsilon,
                attention_fn=attention_fn,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
                sharding=sharding,
            )
        else:
            raise ValueError("pooling_type must be either MAP or CLS.")
        self.position_embeddings = nnx.Param(pos_emb_value, out_sharding=sharding.pos_embed_3d)
        vision_n_positions = n_patches + 1 if self.pooling_type == "CLS" else n_patches
        self.vision_position_ids = nnx.Param(jnp.arange(vision_n_positions, dtype=dtype).reshape(1, -1), out_sharding=sharding.vision_pos_id)

        ln_spec = sharding.layernorm
        if self.use_pre_norm:
            self.ln_pre = nnx.LayerNorm(
                hidden_size,
                epsilon=layernorm_epsilon,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
                scale_init=nnx.with_partitioning(
                    nnx.initializers.ones_init(),
                    ln_spec,
                ),
                bias_init=nnx.with_partitioning(
                    nnx.initializers.zeros_init(),
                    ln_spec,
                ),
            )
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

        _transformer = Transformer(
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            use_quick_gelu=use_quick_gelu,
            use_gradient_checkpointing=use_gradient_checkpointing,
            attention_fn=attention_fn,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            sharding=sharding,
        )
        self.num_layers = num_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.layers = _transformer.layers

        self.ln_post = nnx.LayerNorm(
            hidden_size,
            epsilon=layernorm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            scale_init=nnx.with_partitioning(
                nnx.initializers.ones_init(),
                ln_spec,
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros_init(),
                ln_spec,
            ),
        )

    def __call__(self, img: Float[Array, "batch height width channels"]) -> Float[Array, "batch hidden_size"]:
        """Apply the Vision Transformer to input images.

        Args:
            img (Float[Array, "batch height width channels"]): Batch of input images.

        Returns:
            Float[Array, "batch hidden_size"]: Batch of output embeddings from the pooling
                method ([CLS] token or MultiheadAttentionPooling head).
        """
        patches: Float[Array, "batch patches_h patches_w hidden_size"] = self.patch_embeddings(
            img,
            out_sharding=named_sharding_like(img, P(sharding_of(img).spec[0], None, None, None)),
        )
        batch_size = patches.shape[0]
        patches: Float[Array, "batch n_patches hidden_size"] = patches.reshape(batch_size, -1, patches.shape[-1])
        if self.pooling_type == "CLS":
            cls_token: Float[Array, "batch 1 hidden_size"] = jnp.tile(self.cls_token[...], [batch_size, 1, 1])
            cls_token = reshard_like(cls_token, patches)
            x: Float[Array, "batch n_patches+1 hidden_size"] = jnp.concat([cls_token, patches], axis=1)
        else:
            x: Float[Array, "batch n_patches hidden_size"] = patches
        pos_embed_raw = self.position_embeddings[...]
        pos_embed = jnp.tile(pos_embed_raw, [batch_size, 1, 1])
        pos_embed = reshard_like(pos_embed, x)
        embeddings: Float[Array, "batch length hidden_size"] = x + pos_embed

        if self.use_pre_norm:
            x: Float[Array, "batch length hidden_size"] = self.ln_pre(embeddings)
        else:
            x: Float[Array, "batch length hidden_size"] = self.dropout(embeddings)

        if self.use_gradient_checkpointing:
            x: Float[Array, "batch length hidden_size"] = scan_forward_remat(x, self.layers)
        else:
            x: Float[Array, "batch length hidden_size"] = scan_forward(x, self.layers)
        x: Float[Array, "batch length hidden_size"] = self.ln_post(x)
        if self.pooling_type == "CLS":
            return x[:, 0]
        else:
            return self.map_head(x)
