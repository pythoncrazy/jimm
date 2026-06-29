import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, DTypeLike, Float

from jimm.common.sharding import NoSharding, ShardingSpec, named_sharding_like, reshard_like, sharding_of
from jimm.common.transformer import (
    Transformer,
    rope_cos_sin,
    scan_forward,
    scan_forward_remat,
    scan_forward_rope,
    scan_forward_rope_remat,
)


class MultiHeadAttentionPoolingHead(nnx.Module):
    """Multihead Attention Pooling, as needed by the SigLIP model"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        layernorm_epsilon: float = 1e-6,
        act_fn: Callable | None = None,
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
            act_fn (Callable | None, optional): MLP activation function. When None, defaults to exact GELU.
                Pass ``functools.partial(jax.nn.gelu, approximate=True)`` for SigLIP. Defaults to None.
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

        _mlp_act = act_fn if act_fn is not None else nnx.gelu

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
            _mlp_act,
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
        act_fn: Callable | None = None,
        use_pre_norm: bool = False,
        use_patch_bias: bool = True,
        use_gradient_checkpointing: bool = False,
        use_layer_scale: bool = False,
        layer_scale_init: float = 1.0,
        use_rope: bool = False,
        rope_theta: float = 100.0,
        num_register_tokens: int = 0,
        use_gated_mlp: bool = False,
        key_bias: bool = True,
        attn_bias: bool = True,
        mlp_bias: bool = True,
        use_rms_norm: bool = False,
        pre_norm_before_pos: bool = False,
        attention_fn: Callable[..., Any] | None = None,
        rngs: rnglib.Rngs | None = None,
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        sharding: ShardingSpec = NoSharding,
    ):
        """Initialize the Vision Transformer base model.

        Args:
            img_size (int): The size of the input images. Ignored when use_rope=True (variable size supported).
            patch_size (int): The patch size of the vision transformer.
            in_channels (int): The number of input channels.
            hidden_size (int): The width of the vision transformer.
            num_layers (int): The number of layers in the vision transformer.
            num_heads (int): The number of attention heads in the vision transformer.
            mlp_dim (int): The dimension of the MLP in the transformer blocks.
            layernorm_epsilon (float, optional): Epsilon for LayerNorm or RMSNorm. Defaults to 1e-5.
            pooling_type (str, optional): The pooling method — "CLS", "MAP", or "ALL". "ALL" returns all
                patch token embeddings as (batch, n_patches, hidden_size); no CLS token is prepended in this
                mode. Defaults to "CLS".
            dropout_rate (float, optional): The dropout rate. Defaults to 0.0.
            act_fn (Callable | None, optional): MLP activation function. When None, defaults to exact GELU.
                Pass ``quickgelu`` for CLIP, ``functools.partial(jax.nn.gelu, approximate=True)`` for SigLIP,
                or ``jax.nn.silu`` for SwiGLU-style models. Defaults to None.
            use_pre_norm (bool, optional): Whether to apply a norm after patch+pos embeddings, before the transformer. Defaults to False.
            use_patch_bias (bool, optional): Whether to use bias in the patch embedding convolution. Defaults to True.
            use_gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
            use_layer_scale (bool, optional): Whether to apply per-channel LayerScale to residuals. Defaults to False.
            layer_scale_init (float, optional): Initial value for LayerScale parameters. Defaults to 1.0.
            use_rope (bool, optional): Whether to use 2D rotary position embeddings instead of absolute PE.
                When True, no position_embeddings parameter is created and image size may vary across calls
                (as long as each dimension is divisible by patch_size). Defaults to False.
            rope_theta (float, optional): RoPE base frequency. Only used when use_rope=True. Defaults to 100.0.
            num_register_tokens (int, optional): Number of learnable register tokens prepended between CLS and
                patch tokens. Defaults to 0.
            use_gated_mlp (bool, optional): Whether to use gated (SwiGLU-style) MLP. Defaults to False.
            key_bias (bool, optional): Whether to include a bias in the key projection. Only applies when attn_bias=True. Defaults to True.
            attn_bias (bool, optional): Whether to include biases in all attention projections (q, k, v, out). Defaults to True.
            mlp_bias (bool, optional): Whether to include biases in MLP linear layers. Defaults to True.
            use_rms_norm (bool, optional): Whether to use RMSNorm instead of LayerNorm for all norms. Defaults to False.
            pre_norm_before_pos (bool, optional): When True and use_pre_norm=True, apply ln_pre to patches
                before adding position embeddings (AIMv2 style). When False, apply ln_pre after patch+pos
                addition. Dropout is not applied at the embedding level when use_pre_norm=True regardless
                of this flag. Defaults to False.
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
        if use_rope and use_pre_norm:
            raise ValueError("use_pre_norm is not supported with use_rope=True")
        n_patches: int = (img_size // patch_size) ** 2
        self.use_pre_norm = use_pre_norm
        self.pre_norm_before_pos = pre_norm_before_pos
        self.pooling_type = pooling_type
        self.use_rope = use_rope
        self.patch_size = patch_size
        self.img_size = img_size
        self.rope_theta = rope_theta
        self.head_dim = hidden_size // num_heads
        self.num_register_tokens = num_register_tokens

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
            if num_register_tokens > 0:
                reg_value: Float[Array, "1 num_register_tokens hidden_size"] = nnx.initializers.zeros_init()(rngs.params(), (1, num_register_tokens, hidden_size))
                reg_sharding = getattr(sharding, "register_tokens", sharding.cls_token)
                self.register_tokens = nnx.Param(reg_value, out_sharding=reg_sharding)
            if not use_rope:
                pos_emb_value: Float[Array, "1 n_patches+1 hidden_size"] = nnx.initializers.truncated_normal(stddev=0.02)(rngs.params(), (1, n_patches + 1, hidden_size))
        elif self.pooling_type == "MAP":
            if not use_rope:
                pos_emb_value: Float[Array, "1 n_patches hidden_size"] = nnx.initializers.truncated_normal(stddev=0.02)(rngs.params(), (1, n_patches, hidden_size))
            self.map_head = MultiHeadAttentionPoolingHead(
                hidden_size=hidden_size,
                intermediate_size=4 * hidden_size,
                num_heads=num_heads,
                layernorm_epsilon=layernorm_epsilon,
                act_fn=act_fn,
                attention_fn=attention_fn,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
                sharding=sharding,
            )
        elif self.pooling_type == "ALL":
            if not use_rope:
                pos_emb_value: Float[Array, "1 n_patches hidden_size"] = nnx.initializers.truncated_normal(stddev=0.02)(rngs.params(), (1, n_patches, hidden_size))
        else:
            raise ValueError("pooling_type must be CLS, MAP, or ALL.")
        if not use_rope:
            self.position_embeddings = nnx.Param(pos_emb_value, out_sharding=sharding.pos_embed_3d)
            vision_n_positions = n_patches + 1 if self.pooling_type == "CLS" else n_patches
            self.vision_position_ids = nnx.Param(jnp.arange(vision_n_positions, dtype=dtype).reshape(1, -1), out_sharding=sharding.vision_pos_id)

        ln_spec = sharding.layernorm
        ln_scale_init = nnx.with_partitioning(nnx.initializers.ones_init(), ln_spec)
        if self.use_pre_norm:
            if use_rms_norm:
                self.ln_pre = nnx.RMSNorm(hidden_size, epsilon=layernorm_epsilon, dtype=dtype, param_dtype=param_dtype, rngs=rngs, scale_init=ln_scale_init)
            else:
                self.ln_pre = nnx.LayerNorm(
                    hidden_size,
                    epsilon=layernorm_epsilon,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    rngs=rngs,
                    scale_init=ln_scale_init,
                    bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), ln_spec),
                )
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

        _transformer = Transformer(
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            layernorm_epsilon=layernorm_epsilon,
            dropout_rate=dropout_rate,
            act_fn=act_fn,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_layer_scale=use_layer_scale,
            layer_scale_init=layer_scale_init,
            use_gated_mlp=use_gated_mlp,
            key_bias=key_bias,
            attn_bias=attn_bias,
            mlp_bias=mlp_bias,
            use_rms_norm=use_rms_norm,
            attention_fn=attention_fn,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            sharding=sharding,
        )
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.use_gated_mlp = use_gated_mlp
        self.layernorm_epsilon = layernorm_epsilon
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.layers = _transformer.layers

        if use_rms_norm:
            self.ln_post = nnx.RMSNorm(hidden_size, epsilon=layernorm_epsilon, dtype=dtype, param_dtype=param_dtype, rngs=rngs, scale_init=ln_scale_init)
        else:
            self.ln_post = nnx.LayerNorm(
                hidden_size,
                epsilon=layernorm_epsilon,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
                scale_init=ln_scale_init,
                bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), ln_spec),
            )

    def __call__(self, img: Float[Array, "batch height width channels"]) -> Array:
        """Apply the Vision Transformer to input images.

        Args:
            img (Float[Array, "batch height width channels"]): Batch of input images. When use_rope=True,
                height and width may vary across calls as long as each is divisible by patch_size.

        Returns:
            Array: Output embeddings. Shape depends on pooling_type:
                - "CLS": Float[Array, "batch hidden_size"] — CLS token embedding.
                - "MAP": Float[Array, "batch hidden_size"] — MultiheadAttentionPooling embedding.
                - "ALL": Float[Array, "batch n_patches hidden_size"] — all patch token embeddings.
        """
        if self.use_rope:
            if img.shape[1] % self.patch_size != 0 or img.shape[2] % self.patch_size != 0:
                raise ValueError(f"Image dimensions ({img.shape[1]}, {img.shape[2]}) must each be divisible by patch_size={self.patch_size}")
        patches: Float[Array, "batch patches_h patches_w hidden_size"] = self.patch_embeddings(
            img,
            out_sharding=named_sharding_like(img, P(sharding_of(img).spec[0], None, None, None)),
        )
        batch_size = patches.shape[0]
        patches: Float[Array, "batch n_patches hidden_size"] = patches.reshape(batch_size, -1, patches.shape[-1])
        if self.pooling_type == "CLS":
            cls_token: Float[Array, "batch 1 hidden_size"] = jnp.tile(self.cls_token[...], [batch_size, 1, 1])
            cls_token = reshard_like(cls_token, patches)
            parts = [cls_token]
            if self.num_register_tokens > 0:
                regs: Float[Array, "batch num_register_tokens hidden_size"] = jnp.tile(self.register_tokens[...], [batch_size, 1, 1])
                parts.append(reshard_like(regs, patches))
            parts.append(patches)
            x: Float[Array, "batch n_prefix+n_patches hidden_size"] = jnp.concatenate(parts, axis=1)
        else:
            x: Float[Array, "batch n_patches hidden_size"] = patches

        if self.use_rope:
            x: Float[Array, "batch length hidden_size"] = self.dropout(x)
            cos, sin = rope_cos_sin(img.shape[1], img.shape[2], self.patch_size, self.head_dim, self.rope_theta)
            if self.use_gradient_checkpointing:
                (x, _) = scan_forward_rope_remat((x, (cos, sin)), self.layers)
            else:
                (x, _) = scan_forward_rope((x, (cos, sin)), self.layers)
        else:
            pos_embed_raw = self.position_embeddings[...]
            pos_embed = jnp.tile(pos_embed_raw, [batch_size, 1, 1])
            pos_embed = reshard_like(pos_embed, x)
            if self.use_pre_norm and self.pre_norm_before_pos:
                x: Float[Array, "batch length hidden_size"] = self.ln_pre(x) + pos_embed
            else:
                x = x + pos_embed
                if self.use_pre_norm:
                    x: Float[Array, "batch length hidden_size"] = self.ln_pre(x)
                else:
                    x: Float[Array, "batch length hidden_size"] = self.dropout(x)

            if self.use_gradient_checkpointing:
                x: Float[Array, "batch length hidden_size"] = scan_forward_remat(x, self.layers)
            else:
                x: Float[Array, "batch length hidden_size"] = scan_forward(x, self.layers)

        x: Float[Array, "batch length hidden_size"] = self.ln_post(x)
        if self.pooling_type == "CLS":
            return x[:, 0]
        elif self.pooling_type == "ALL":
            return x
        else:
            return self.map_head(x)
