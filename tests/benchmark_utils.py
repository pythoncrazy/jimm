from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

import jax

AUTOTUNE_CACHE_DIR = os.path.join(os.path.dirname(__file__), "tokamax_cache")


def peak_hbm_mb(fn: Any, *args: Any) -> float:
    """Return peak HBM in MB, measured synchronously after a forward pass.

    Calls ``fn(*args)``, blocks until ready, then reads ``bytes_in_use`` across
    all devices. At this point the output and parameters are still live, giving
    the full resident footprint seen by ``tpu-info``.

    Args:
        fn (Any): Callable to run.
        *args (Any): Arguments forwarded to ``fn``.

    Returns:
        float: Max ``bytes_in_use`` across devices in MB.
    """
    jax.block_until_ready(fn(*args))
    return max((d.memory_stats() or {}).get("bytes_in_use", 0) for d in jax.devices()) / 1024**2


def bench(
    fn: Callable[..., Any],
    *args: Any,
    n_warmup: int = 3,
    n_runs: int = 10,
) -> float:
    """Benchmark a callable, returning mean wall-clock latency in ms.

    Args:
        fn (Callable[..., Any]): Callable to benchmark.
        *args (Any): Arguments forwarded to ``fn`` on every call.
        n_warmup (int): Warmup calls before timing. Defaults to 3.
        n_runs (int): Timed calls. Defaults to 10.

    Returns:
        float: Mean wall-clock time per call in milliseconds.
    """
    for _ in range(n_warmup):
        jax.block_until_ready(fn(*args))
    t0 = time.perf_counter()
    for _ in range(n_runs):
        jax.block_until_ready(fn(*args))
    return (time.perf_counter() - t0) / n_runs * 1000
