from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, Literal, cast

import jax.numpy as jnp
from jax._src.mesh import get_concrete_mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int
from tokamax._src.ops.attention import api as _tokamax_api

_SPLASH_BLOCK = 128

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
    q_seq = query.shape[1]
    kv_seq = key.shape[1]
    q_pad = (-q_seq) % _SPLASH_BLOCK
    kv_pad = (-kv_seq) % _SPLASH_BLOCK

    q_seq_lengths: Int[Array, " batch"] | None = None
    kv_seq_lengths: Int[Array, " batch"] | None = None

    if q_pad > 0:
        query = jnp.pad(query, ((0, 0), (0, q_pad), (0, 0), (0, 0)))
        q_seq_lengths = jnp.full(query.shape[:1], q_seq)
    if kv_pad > 0:
        key = jnp.pad(key, ((0, 0), (0, kv_pad), (0, 0), (0, 0)))
        value = jnp.pad(value, ((0, 0), (0, kv_pad), (0, 0), (0, 0)))
        kv_seq_lengths = jnp.full(key.shape[:1], kv_seq)

    q_sharding: NamedSharding | None = None
    mesh = get_concrete_mesh()
    if mesh is not None and mesh.size > 1:
        batch = query.shape[0]
        spec = P(tuple(mesh.axis_names), None, None, None) if batch % mesh.size == 0 else P()
        q_sharding = NamedSharding(mesh, spec)

    out = _tokamax_api.dot_product_attention(
        query,
        key,
        value,
        bias=bias,
        mask=mask,
        is_causal=causal,
        query_seq_lengths=q_seq_lengths,
        key_value_seq_lengths=kv_seq_lengths,
        implementation=cast(Any, implementation),
        q_sharding=q_sharding,
    )
    return out[:, :q_seq] if q_pad > 0 else out


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
