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


def rope_cos_sin(
    img_h: int,
    img_w: int,
    patch_size: int,
    head_dim: int,
    rope_theta: float,
) -> tuple[Float[Array, "n_patches head_dim"], Float[Array, "n_patches head_dim"]]:
    """Compute 2D RoPE cos/sin from image spatial dimensions.

    Args:
        img_h (int): Image height in pixels.
        img_w (int): Image width in pixels.
        patch_size (int): Patch size in pixels.
        head_dim (int): Attention head dimension.
        rope_theta (float): RoPE base frequency.

    Returns:
        tuple[Float[Array, "n_patches head_dim"], Float[Array, "n_patches head_dim"]]:
            cos and sin tensors, each shape (n_h*n_w, head_dim).
    """
    n_h, n_w = img_h // patch_size, img_w // patch_size
    ch = (jnp.arange(0.5, n_h, dtype=jnp.float32) / n_h) * 2 - 1
    cw = (jnp.arange(0.5, n_w, dtype=jnp.float32) / n_w) * 2 - 1
    coords = jnp.stack(jnp.meshgrid(ch, cw, indexing="ij"), axis=-1).reshape(-1, 2)
    inv_freq = 1.0 / rope_theta ** jnp.arange(0.0, 1.0, 4.0 / head_dim, dtype=jnp.float32)
    angles = 2 * jnp.pi * coords[:, :, None] * inv_freq[None, None, :]
    angles = jnp.tile(angles.reshape(n_h * n_w, -1), (1, 2))
    return jnp.cos(angles), jnp.sin(angles)


def _rotate_half(x: Float[Array, "... d"]) -> Float[Array, "... d"]:
    """Rotate the last dimension by half for RoPE application.

    Args:
        x (Float[Array, "... d"]): Input tensor.

    Returns:
        Float[Array, "... d"]: Tensor with last dim rotated: concat(-x[d/2:], x[:d/2]).
    """
    d = x.shape[-1] // 2
    return jnp.concatenate([-x[..., d:], x[..., :d]], axis=-1)


def apply_rope(
    q: Float[Array, "batch heads seq head_dim"],
    k: Float[Array, "batch heads seq head_dim"],
    cos: Float[Array, "n_patches head_dim"],
    sin: Float[Array, "n_patches head_dim"],
) -> tuple[
    Float[Array, "batch heads seq head_dim"],
    Float[Array, "batch heads seq head_dim"],
]:
    """Apply RoPE to Q and K for patch tokens only; prefix tokens are unchanged.

    Args:
        q (Float[Array, "batch heads seq head_dim"]): Query tensor.
        k (Float[Array, "batch heads seq head_dim"]): Key tensor.
        cos (Float[Array, "n_patches head_dim"]): RoPE cosines for patch positions.
        sin (Float[Array, "n_patches head_dim"]): RoPE sines for patch positions.

    Returns:
        tuple[...]: Rotated q and k with prefix tokens unmodified.
    """
    n_pre = q.shape[2] - cos.shape[0]
    if n_pre < 0:
        raise ValueError(f"cos has {cos.shape[0]} patch positions but q only has {q.shape[2]} tokens")
    c, s = cos[None, None], sin[None, None]

    def rot(x: Array) -> Array:
        return x * c + _rotate_half(x) * s

    return (
        jnp.concatenate([q[:, :, :n_pre], rot(q[:, :, n_pre:])], axis=2),
        jnp.concatenate([k[:, :, :n_pre], rot(k[:, :, n_pre:])], axis=2),
    )


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
        use_layer_scale: bool = False,
        layer_scale_init: float = 1.0,
        use_gated_mlp: bool = False,
        hidden_act: str = "gelu",
        key_bias: bool = True,
        attn_bias: bool = True,
        mlp_bias: bool = True,
        use_rms_norm: bool = False,
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
            use_layer_scale (bool, optional): Whether to apply per-channel LayerScale to residuals (DINOv2). Defaults to False.
            layer_scale_init (float, optional): Initial value for LayerScale parameters. Defaults to 1.0.
                When training from scratch the DINOv2 paper uses much smaller values (e.g. 1e-5 for large models).
            use_gated_mlp (bool, optional): Whether to use gated (SwiGLU-style) MLP. Defaults to False.
            hidden_act (str, optional): Activation function for (gated) MLP — "gelu" or "silu". Defaults to "gelu".
            key_bias (bool, optional): Whether to include a bias in the key projection. Only applies when attn_bias=True. Defaults to True.
            attn_bias (bool, optional): Whether to include biases in all attention projections (q, k, v, out). Defaults to True.
            mlp_bias (bool, optional): Whether to include biases in MLP linear layers. Defaults to True.
            use_rms_norm (bool, optional): Whether to use RMSNorm instead of LayerNorm for norm1 and norm2. Defaults to False.
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
        if hidden_act not in ("gelu", "silu"):
            raise ValueError(f"hidden_act must be 'gelu' or 'silu', got {hidden_act!r}")
        self.attn_mask = nnx.Variable(attn_mask) if attn_mask is not None else None
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_layer_scale = use_layer_scale
        self.use_gated_mlp = use_gated_mlp
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self._skip_flax_mask = attention_fn is not None
        self._act_fn = jax.nn.silu if hidden_act == "silu" else functools.partial(jax.nn.gelu, approximate=False)
        if use_layer_scale:
            ls_init: Float[Array, " hidden_size"] = jnp.full((hidden_size,), layer_scale_init, dtype=param_dtype)
            self.layer_scale1 = nnx.Param(ls_init, out_sharding=sharding.layer_scale)
            self.layer_scale2 = nnx.Param(ls_init, out_sharding=sharding.layer_scale)
        ln_init = nnx.with_partitioning(nnx.initializers.ones_init(), sharding.layernorm)
        if use_rms_norm:
            self.norm1 = nnx.RMSNorm(hidden_size, epsilon=layernorm_epsilon, dtype=dtype, param_dtype=param_dtype, rngs=rngs, scale_init=ln_init)
        else:
            self.norm1 = nnx.LayerNorm(
                hidden_size,
                epsilon=layernorm_epsilon,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
                scale_init=ln_init,
                bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), sharding.layernorm),
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
            use_bias=attn_bias,
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
        if not key_bias and attn_bias:
            # Replace key projection with one that never had a bias; avoids
            # post-construction mutation of NNX variable state.
            self.attn.key = nnx.LinearGeneral(
                in_features=hidden_size,
                out_features=(num_heads, self.head_dim),
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
                kernel_init=nnx.with_partitioning(
                    nnx.initializers.xavier_uniform(),
                    sharding.attn_qkv_kernel,
                ),
            )
        if use_rms_norm:
            self.norm2 = nnx.RMSNorm(hidden_size, epsilon=layernorm_epsilon, dtype=dtype, param_dtype=param_dtype, rngs=rngs, scale_init=ln_init)
        else:
            self.norm2 = nnx.LayerNorm(
                hidden_size,
                epsilon=layernorm_epsilon,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
                scale_init=ln_init,
                bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), sharding.layernorm),
            )

        # nnx.gelu == jax.nn.gelu(approximate=False); explicit here so DINOv2's exact-GELU requirement is clear.
        activation_fn = quickgelu if use_quick_gelu else functools.partial(jax.nn.gelu, approximate=False)

        def _lin(in_f: int, out_f: int, k_spec: Any, b_spec: Any) -> nnx.Linear:
            return nnx.Linear(
                in_f,
                out_f,
                use_bias=mlp_bias,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
                kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), k_spec),
                bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), b_spec),
            )

        if use_gated_mlp:
            self.gate = _lin(hidden_size, mlp_dim, sharding.mlp_up_kernel, sharding.mlp_up_bias)
            self.up = _lin(hidden_size, mlp_dim, sharding.mlp_up_kernel, sharding.mlp_up_bias)
            self.down = _lin(mlp_dim, hidden_size, sharding.mlp_down_kernel, sharding.mlp_down_bias)
            self.gated_dropout = nnx.Dropout(dropout_rate, rngs=rngs)
        else:
            self.mlp = nnx.Sequential(
                _lin(hidden_size, mlp_dim, sharding.mlp_up_kernel, sharding.mlp_up_bias),
                activation_fn,
                nnx.Dropout(dropout_rate, rngs=rngs),
                _lin(mlp_dim, hidden_size, sharding.mlp_down_kernel, sharding.mlp_down_bias),
                nnx.Dropout(dropout_rate, rngs=rngs),
            )

    def _mlp(self, x: Float[Array, "batch seq hidden"]) -> Float[Array, "batch seq hidden"]:
        """Apply MLP sublayer (standard sequential or gated SwiGLU).

        Args:
            x (Float[Array, "batch seq hidden"]): Input tensor.

        Returns:
            Float[Array, "batch seq hidden"]: MLP output.
        """
        if self.use_gated_mlp:
            return self.down(self.gated_dropout(self._act_fn(self.gate(x)) * self.up(x)))
        return self.mlp(x)

    def __call__(
        self,
        x: Float[Array, "batch seq hidden"],
        pos_emb: tuple[Float[Array, "n_patches head_dim"], Float[Array, "n_patches head_dim"]] | None = None,
    ) -> Float[Array, "batch seq hidden"]:
        """Apply the transformer encoder to the input.

        Args:
            x (Float[Array, "batch seq hidden"]): Input tensor with shape [batch, sequence_length, hidden_size].
            pos_emb (tuple[Array, Array] | None, optional): RoPE (cos, sin) tensors, each (n_patches, head_dim).
                When provided, RoPE is applied to Q and K via the attention sub-projections directly (bypassing
                nnx.MultiHeadAttention.__call__), so attention_fn and attention dropout are not applied.
                Defaults to None.

        Returns:
            Float[Array, "batch seq hidden"]: Output tensor with the same shape as input.
        """
        if pos_emb is not None:
            normed = self.norm1(x)
            batch, seq, _ = normed.shape
            q = self.attn.query(normed).reshape(batch, seq, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            k = self.attn.key(normed).reshape(batch, seq, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            v = self.attn.value(normed).reshape(batch, seq, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            q, k = apply_rope(q, k, *pos_emb)
            scale = self.head_dim**-0.5
            attn_w = jax.nn.softmax(jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale, axis=-1)
            out = jnp.einsum("bhqk,bhkd->bhqd", attn_w, v).transpose(0, 2, 1, 3)
            attn_out = reshard_like(self.attn.out(out), x)
            x = x + (self.layer_scale1[...] * attn_out if self.use_layer_scale else attn_out)
            mlp_out = reshard_like(self._mlp(self.norm2(x)), x)
            return x + (self.layer_scale2[...] * mlp_out if self.use_layer_scale else mlp_out)

        seq_len = x.shape[1]
        mask = None
        if self.attn_mask is not None and not self._skip_flax_mask:
            attn_mask_val = self.attn_mask[...]
            mask_seq_len = min(seq_len, attn_mask_val.shape[0])
            mask = attn_mask_val[:mask_seq_len, :mask_seq_len]

        if self.use_gradient_checkpointing:
            attn_out = jax.checkpoint(lambda hidden: self.attn(self.norm1(hidden), mask=mask))(x)
            attn_out = reshard_like(attn_out, x)
            x = x + (self.layer_scale1[...] * attn_out if self.use_layer_scale else attn_out)
            mlp_out = jax.checkpoint(lambda hidden: self._mlp(self.norm2(hidden)))(x)
            mlp_out = reshard_like(mlp_out, x)
            return x + (self.layer_scale2[...] * mlp_out if self.use_layer_scale else mlp_out)

        attn_out = reshard_like(self.attn(self.norm1(x), mask=mask), x)
        x = x + (self.layer_scale1[...] * attn_out if self.use_layer_scale else attn_out)
        mlp_out = reshard_like(self._mlp(self.norm2(x)), x)
        return x + (self.layer_scale2[...] * mlp_out if self.use_layer_scale else mlp_out)


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


@nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
def scan_forward_rope(
    carry: tuple[
        Float[Array, "batch seq hidden"],
        tuple[Float[Array, "n_patches head_dim"], Float[Array, "n_patches head_dim"]],
    ],
    layer: TransformerEncoder,
) -> tuple[
    Float[Array, "batch seq hidden"],
    tuple[Float[Array, "n_patches head_dim"], Float[Array, "n_patches head_dim"]],
]:
    """Apply a single TransformerEncoder layer with RoPE inside an nnx.scan loop.

    Args:
        carry (tuple): (x, pos_emb) where pos_emb = (cos, sin) from rope_cos_sin.
        layer (TransformerEncoder): Batched layer module scanned over axis 0.

    Returns:
        tuple: Updated (x, pos_emb) carry after applying ``layer`` with RoPE.
    """
    x, pos_emb = carry
    return layer(x, pos_emb), pos_emb


@nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
def scan_forward_rope_remat(
    carry: tuple[
        Float[Array, "batch seq hidden"],
        tuple[Float[Array, "n_patches head_dim"], Float[Array, "n_patches head_dim"]],
    ],
    layer: TransformerEncoder,
) -> tuple[
    Float[Array, "batch seq hidden"],
    tuple[Float[Array, "n_patches head_dim"], Float[Array, "n_patches head_dim"]],
]:
    """Apply a single TransformerEncoder layer with RoPE and gradient checkpointing inside an nnx.scan loop.

    Args:
        carry (tuple): (x, pos_emb) where pos_emb = (cos, sin) from rope_cos_sin.
        layer (TransformerEncoder): Batched layer module scanned over axis 0.

    Returns:
        tuple: Updated (x, pos_emb) carry after applying ``layer`` with rematerialization and RoPE.
    """
    x, pos_emb = carry
    return jax.checkpoint(lambda h: layer(h, pos_emb))(x), pos_emb


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
        use_layer_scale: bool = False,
        layer_scale_init: float = 1.0,
        use_gated_mlp: bool = False,
        hidden_act: str = "gelu",
        key_bias: bool = True,
        attn_bias: bool = True,
        mlp_bias: bool = True,
        use_rms_norm: bool = False,
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
            use_layer_scale (bool, optional): Whether to apply per-channel LayerScale to residuals. Defaults to False.
            layer_scale_init (float, optional): Initial value for LayerScale parameters. Defaults to 1.0.
                When training from scratch the DINOv2 paper uses much smaller values (e.g. 1e-5 for large models).
            use_gated_mlp (bool, optional): Whether to use gated (SwiGLU-style) MLP. Defaults to False.
            hidden_act (str, optional): Activation for (gated) MLP — "gelu" or "silu". Defaults to "gelu".
            key_bias (bool, optional): Whether to include a bias in the key projection. Only applies when attn_bias=True. Defaults to True.
            attn_bias (bool, optional): Whether to include biases in all attention projections. Defaults to True.
            mlp_bias (bool, optional): Whether to include biases in MLP linear layers. Defaults to True.
            use_rms_norm (bool, optional): Whether to use RMSNorm instead of LayerNorm. Defaults to False.
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
                use_layer_scale=use_layer_scale,
                layer_scale_init=layer_scale_init,
                use_gated_mlp=use_gated_mlp,
                hidden_act=hidden_act,
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
