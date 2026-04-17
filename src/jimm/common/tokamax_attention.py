from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, Literal, cast

from jaxtyping import Array, Float
from tokamax._src.ops.attention import api as _tokamax_api

Implementation = Literal["mosaic", "triton", "cudnn", "xla", "xla_chunked"]


def tokamax_attention_fn(
    query: Float[Array, "batch q_seq heads head_dim"],
    key: Float[Array, "batch kv_seq heads head_dim"],
    value: Float[Array, "batch kv_seq heads head_dim"],
    bias: Float[Array, "batch heads q_seq kv_seq"] | None = None,
    mask: Array | None = None,
    *,
    causal: bool = False,
    implementation: Implementation | list[str] | None = None,
    **_: Any,
) -> Float[Array, "batch q_seq heads head_dim"]:
    """Wraps ``tokamax.dot_product_attention`` for ``nnx.MultiHeadAttention``.

    Passes inputs in Flax layout ``[batch, seq, heads, head_dim]`` directly —
    no transposition needed. Scale ``1/sqrt(head_dim)`` is computed
    automatically. Auto-selects the best available backend when
    ``implementation`` is ``None`` (mosaic → triton → xla on GPU;
    mosaic_tpu → xla on TPU).

    Args:
        query (Float[Array, "batch q_seq heads head_dim"]): Query tensor.
        key (Float[Array, "batch kv_seq heads head_dim"]): Key tensor.
        value (Float[Array, "batch kv_seq heads head_dim"]): Value tensor.
        bias (Float[Array, "batch heads q_seq kv_seq"] | None): Additive bias.
        mask (Array | None): Boolean mask; ``True`` means attend.
        causal (bool): Apply causal masking. Defaults to ``False``.
        implementation (Implementation | list[Implementation] | None): Backend
            or ordered fallback list. ``None`` auto-selects. Defaults to
            ``None``.
        **_: Absorbs unused kwargs from ``nnx.MultiHeadAttention``.

    Returns:
        Float[Array, "batch q_seq heads head_dim"]: Attention output.
    """
    return _tokamax_api.dot_product_attention(
        query, key, value, bias=bias, mask=mask, is_causal=causal,
        implementation=cast(Any, implementation),
    )


def make_tokamax_attention(
    implementation: Implementation | list[str] | None = None,
) -> Callable[..., Any]:
    """Return an ``attention_fn`` bound to a specific tokamax backend.

    Args:
        implementation (Implementation | list[Implementation] | None): Backend
            name or ordered fallback list. When a list is given, the first
            backend that does not raise ``NotImplementedError`` is used —
            useful for cross-hardware portability, e.g.
            ``["mosaic_tpu", "xla_chunked"]``. ``None`` auto-selects.
            Defaults to ``None``.

    Returns:
        Callable[..., Any]: Partial of :func:`tokamax_attention_fn` with
        ``implementation`` bound.
    """
    return functools.partial(tokamax_attention_fn, implementation=implementation)
