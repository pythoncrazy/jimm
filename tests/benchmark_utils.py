from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from typing import Any

import jax

AUTOTUNE_CACHE_DIR = os.path.join(os.path.dirname(__file__), "tokamax_cache")


def _tensors_mb() -> float:
    return max((d.memory_stats() or {}).get("bytes_in_use", 0) for d in jax.devices()) / 1024**2


def peak_hbm_mb(fn: Any, *args: Any, poll_interval_s: float = 0.005) -> float:
    """Return peak HBM in MB during a forward pass, measured by polling bytes_in_use.

    Spawns a background thread that polls ``bytes_in_use`` every ``poll_interval_s``
    seconds while ``fn(*args)`` executes, capturing the true peak including transient
    activations that may be freed before the call returns.

    Args:
        fn (Any): Callable to run.
        *args (Any): Arguments forwarded to ``fn``.
        poll_interval_s (float): Polling interval in seconds. Defaults to 5ms.

    Returns:
        float: Peak ``bytes_in_use`` across devices in MB.
    """
    peak: list[float] = [0.0]
    stop = threading.Event()

    def _poll() -> None:
        while not stop.is_set():
            v = _tensors_mb()
            if v > peak[0]:
                peak[0] = v
            time.sleep(poll_interval_s)

    thr = threading.Thread(target=_poll, daemon=True)
    thr.start()
    jax.block_until_ready(fn(*args))
    stop.set()
    thr.join()
    return peak[0]


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
