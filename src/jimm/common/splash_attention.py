"""Splash Attention integration for TPU-optimized attention."""

import importlib.util
from dataclasses import dataclass
from typing import Callable, Literal

import jax
from jaxtyping import Array, Float

_TOKAMAX_AVAILABLE = importlib.util.find_spec("tokamax") is not None

if _TOKAMAX_AVAILABLE:
    from tokamax._src.ops.experimental.tpu.splash_attention import (
        splash_attention_kernel as splash,
    )
    from tokamax._src.ops.experimental.tpu.splash_attention import (
        splash_attention_mask as mask_lib,
    )


@dataclass
class SplashAttentionConfig:
    """Configuration for splash attention.

    Attributes:
        enabled (bool): Whether to enable splash attention.
        mask_type (Literal["full", "causal"]): Type of attention mask.
        block_q (int): Block size for query sequence tiling.
        block_kv (int): Block size for key/value sequence tiling.
    """

    enabled: bool = False
    mask_type: Literal["full", "causal"] = "full"
    block_q: int = 128
    block_kv: int = 128


_kernel_cache: dict[tuple[int, int, int, str, int, int], Callable] = {}


def _create_splash_kernel(
    seq_len: int,
    num_heads: int,
    head_dim: int,
    config: SplashAttentionConfig,
) -> Callable[
    [Float[Array, "heads seq head_dim"], Float[Array, "heads seq head_dim"], Float[Array, "heads seq head_dim"]],
    Float[Array, "heads seq head_dim"],
]:
    """Create a cached splash attention kernel.

    Args:
        seq_len (int): Sequence length.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimension of each attention head.
        config (SplashAttentionConfig): Splash attention configuration.

    Returns:
        Callable: A splash attention kernel function.
    """
    cache_key = (seq_len, num_heads, head_dim, config.mask_type, config.block_q, config.block_kv)
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]

    mask_shape = (seq_len, seq_len)
    mask = mask_lib.CausalMask(mask_shape) if config.mask_type == "causal" else mask_lib.FullMask(mask_shape)

    splash_config = splash.SplashConfig(
        block_q=config.block_q,
        block_kv=config.block_kv,
        block_kv_compute=config.block_kv,
        block_q_dkv=config.block_q,
        block_kv_dkv=config.block_kv,
        block_kv_dkv_compute=config.block_kv,
    )

    kernel = splash.make_splash_mha_single_device(mask=mask, config=splash_config)
    _kernel_cache[cache_key] = kernel
    return kernel


def create_splash_attention_fn(
    config: SplashAttentionConfig,
    num_heads: int,
    head_dim: int,
) -> Callable[..., Float[Array, "batch heads seq head_dim"]]:
    """Create a splash attention function compatible with nnx.MultiHeadAttention.

    Args:
        config (SplashAttentionConfig): Splash attention configuration.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimension of each attention head.

    Returns:
        Callable: An attention function. Returns splash attention if enabled and available,
            otherwise returns the default dot_product_attention.
    """
    if not _TOKAMAX_AVAILABLE or not config.enabled:
        from flax.nnx.nn.attention import dot_product_attention

        return dot_product_attention

    def splash_attention_fn(
        query: Float[Array, "batch heads seq head_dim"],
        key: Float[Array, "batch heads seq head_dim"],
        value: Float[Array, "batch heads seq head_dim"],
    ) -> Float[Array, "batch heads seq head_dim"]:
        """Splash attention function.

        Args:
            query (Float[Array, "batch heads seq head_dim"]): Query tensor.
            key (Float[Array, "batch heads seq head_dim"]): Key tensor.
            value (Float[Array, "batch heads seq head_dim"]): Value tensor.

        Returns:
            Float[Array, "batch heads seq head_dim"]: Output tensor.
        """
        seq_len = query.shape[2]
        kernel = _create_splash_kernel(seq_len, num_heads, head_dim, config)
        return jax.vmap(kernel)(query, key, value)

    return splash_attention_fn
