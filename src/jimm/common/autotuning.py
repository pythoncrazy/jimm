from __future__ import annotations

import contextlib
import functools
import hashlib
import os
from collections.abc import Callable
from typing import Any

from jax.extend import backend as _jax_backend
from tokamax._src.autotuning import api as _autotune_api

AutotuningResult = _autotune_api.AutotuningResult


def _lower(jitted_fn: Any, *args: Any) -> Any:
    """Unwrap a Flax ``nnx.jit`` ``Lowered`` to the raw ``jax.stages.Lowered``.

    ``nnx.jit`` wraps the JAX lowered object; tokamax's HLO utils require the
    raw type, accessible via the ``.lowered`` attribute.

    Args:
        jitted_fn (Any): An ``nnx.jit`` or ``jax.jit`` callable.
        *args (Any): Sample arguments for lowering.

    Returns:
        Any: Raw ``jax.stages.Lowered`` computation.
    """
    lowered = jitted_fn.lower(*args)
    return getattr(lowered, "lowered", lowered)


def autotune(
    jitted_fn: Any,
    *sample_args: Any,
    save_path: str | os.PathLike | None = None,
    **kwargs: Any,
) -> AutotuningResult:
    """Autotune all tokamax ops in a jitted function and optionally save results.

    Lowers ``jitted_fn`` with ``sample_args`` to extract op shapes, then
    microbenchmarks every kernel config. Use the result as a context manager::

        result = jimm.autotune(forward, model, image, text)
        with result:
            out = forward(model, image, text)

    Args:
        jitted_fn (Any): Jitted callable containing tokamax ops.
        *sample_args (Any): Representative inputs; shapes/dtypes must match
            production inputs.
        save_path (str | os.PathLike | None): Serialize result as JSON here.
        **kwargs (Any): Forwarded to ``tokamax.autotune``
            (e.g. ``all_implementations``, ``progress_bar``).

    Returns:
        AutotuningResult: Best config for each op found in ``jitted_fn``.
    """
    result = _autotune_api.autotune(_lower(jitted_fn, *sample_args), **kwargs)
    if save_path is not None:
        with open(save_path, "w") as f:
            result.dump(f)
    return result


def load_autotune_result(path: str | os.PathLike) -> AutotuningResult:
    """Load an :class:`AutotuningResult` from a JSON file.

    Args:
        path (str | os.PathLike): Path written by :func:`autotune` or
            :func:`cached_autotune`.

    Returns:
        AutotuningResult: Deserialized result.
    """
    with open(path) as f:
        return AutotuningResult.load(f)


def cached_autotune(
    jitted_fn: Any,
    *sample_args: Any,
    cache_dir: str | os.PathLike,
    **kwargs: Any,
) -> AutotuningResult | None:
    """Load autotuning results from disk, or run autotune and cache them.

    The cache key is an MD5 hash of op autotuning keys + device kind, so the
    same entry is reused whenever model architecture and hardware are unchanged.

    Args:
        jitted_fn (Any): Jitted callable containing tokamax ops.
        *sample_args (Any): Representative inputs used to extract op shapes.
        cache_dir (str | os.PathLike): Directory for JSON cache files.
        **kwargs (Any): Forwarded to ``tokamax.autotune``.

    Returns:
        AutotuningResult | None: Loaded or freshly computed result, or ``None``
        if no tunable ops exist.
    """
    bound_args = _autotune_api.get_bound_args(_lower(jitted_fn, *sample_args))
    if not bound_args:
        return None

    device_kind = _jax_backend.get_default_device().device_kind
    key_parts = sorted(str(ba.autotuning_cache_key) for ba in bound_args)
    cache_key = hashlib.md5(
        (device_kind + ":" + "|".join(key_parts)).encode()
    ).hexdigest()[:16]

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{cache_key}.json")

    if os.path.exists(cache_path):
        return load_autotune_result(cache_path)

    result = _autotune_api.autotune(list(bound_args), **kwargs)
    with open(cache_path, "w") as f:
        result.dump(f)
    return result


def autotuned_fn(
    jitted_fn: Any,
    *sample_args: Any,
    cache_dir: str | os.PathLike,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Wrap a jitted function to always run with tuned tokamax configs.

    Eagerly calls :func:`cached_autotune`, then returns a wrapper that injects
    the tuned configs via :class:`AutotuningResult` on every call.

    Args:
        jitted_fn (Any): Jitted callable containing tokamax ops.
        *sample_args (Any): Representative inputs used for autotuning.
        cache_dir (str | os.PathLike): Directory for the autotune cache.
        **kwargs (Any): Forwarded to ``tokamax.autotune``.

    Returns:
        Callable[..., Any]: Wrapped callable that applies tuned configs on
        every invocation.
    """
    result = cached_autotune(jitted_fn, *sample_args, cache_dir=cache_dir, **kwargs)
    ctx = result if result is not None else contextlib.nullcontext()

    @functools.wraps(jitted_fn)
    def wrapper(*args: Any, **kw: Any) -> Any:
        with ctx:
            return jitted_fn(*args, **kw)

    return wrapper
