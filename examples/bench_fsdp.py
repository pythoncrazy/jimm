"""Benchmark FSDP scaling: 1, 2, 4-way sharding with jax.jit (nnx.split/merge).

Shows how tensor memory and throughput scale as you add FSDP devices,
keeping batch_per_device=32 constant so total batch grows with device count.

Memory columns:
  fwd_tens / bwd_tens  - peak bytes_in_use during fwd/bwd (polled at 5ms)
  total_hbm            - bytes_limit - largest_free_block at end (tensors + XLA executables, cumulative)
"""

import threading
import time
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax.experimental import mesh_utils
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from transformers import AutoConfig

from jimm import CLIP

HF_MODEL_NAME = "openai/clip-vit-large-patch14"
config = AutoConfig.from_pretrained(HF_MODEL_NAME).to_dict()
text_config = config["text_config"]
vision_config = config["vision_config"]

BATCH_PER_DEVICE = 8


def clip_loss(logits: jax.Array, mesh: Mesh) -> jax.Array:
    logits = jax.sharding.reshard(logits, NamedSharding(mesh, P(None, None)))
    labels = jnp.arange(logits.shape[0])
    loss_i = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    loss_t = optax.softmax_cross_entropy_with_integer_labels(logits.T, labels)
    return (loss_i.mean() + loss_t.mean()) / 2


def tensors_mb() -> float:
    return max((d.memory_stats() or {}).get("bytes_in_use", 0) for d in jax.devices()) / 1024**2


def total_hbm_mb() -> float:
    return max((s.get("bytes_limit", 0) - s.get("largest_free_block_bytes", 0)) for d in jax.devices() if (s := d.memory_stats() or {})) / 1024**2


def bench_ms(fn, *args, n_warmup: int = 3, n_runs: int = 10) -> tuple[float, float]:
    for _ in range(n_warmup):
        jax.block_until_ready(fn(*args))
    peak = [0.0]
    stop = threading.Event()

    def _poll():
        while not stop.is_set():
            v = tensors_mb()
            if v > peak[0]:
                peak[0] = v
            time.sleep(0.005)

    thr = threading.Thread(target=_poll, daemon=True)
    thr.start()
    jax.block_until_ready(fn(*args))
    stop.set()
    thr.join()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append((time.perf_counter() - t0) * 1000)
    return sum(times) / len(times), peak[0]


def run(n_fsdp: int):
    batch = BATCH_PER_DEVICE * n_fsdp

    all_devices = jax.devices()
    dev_grid = mesh_utils.create_device_mesh((1, n_fsdp), devices=all_devices[:n_fsdp])
    mesh = Mesh(dev_grid, ("data", "fsdp"), axis_types=(AxisType.Explicit, AxisType.Explicit))
    jax.set_mesh(mesh)

    img = jax.device_put(
        jnp.ones((batch, vision_config["image_size"], vision_config["image_size"], 3)),
        NamedSharding(mesh, P("fsdp", None, None, None)),
    )
    txt = jax.device_put(
        jnp.ones((batch, text_config["max_position_embeddings"]), dtype=jnp.int32),
        NamedSharding(mesh, P("fsdp", None)),
    )

    model = CLIP.from_config(config, rngs=nnx.Rngs(0))
    model.eval()
    optimizer = nnx.Optimizer(model, optax.adamw(1e-4), wrt=nnx.Param)

    graphdef, state = nnx.split((model, optimizer))

    @partial(jax.jit)
    def forward(state: nnx.State, image: jax.Array, text: jax.Array) -> jax.Array:
        m, _ = nnx.merge(graphdef, state)
        return m(image, text)

    @partial(jax.jit)
    def train_step(state: nnx.State, image: jax.Array, text: jax.Array) -> nnx.State:
        m, opt = nnx.merge(graphdef, state)

        def loss_fn(m: CLIP) -> jax.Array:
            return clip_loss(m(image, text), mesh)

        grads = nnx.grad(loss_fn)(m)
        opt.update(m, grads)
        return nnx.state((m, opt))

    fwd_ms, fwd_tens = bench_ms(forward, state, img, txt)
    bwd_ms, bwd_tens = bench_ms(train_step, state, img, txt)
    hbm = total_hbm_mb()

    row = f"fsdp={n_fsdp}  bs={batch} ({BATCH_PER_DEVICE}/dev)"
    print(f"{row:32s} {fwd_tens:>8.1f}M {bwd_tens:>8.1f}M {hbm:>9.1f}M {fwd_ms:>7.1f}ms {bwd_ms:>10.1f}ms")


n_available = jax.device_count()
print(f"Available devices: {n_available}")
print(f"{'':32s} {'fwd_tens':>8} {'bwd_tens':>8} {'total_hbm':>9} {'fwd ms':>8} {'fwd+bwd ms':>11}")
print(f"{'':32s} {'(polled)':>8} {'(polled)':>8} {'(cumul)':>9}")

for n in [1, 2, 4]:
    if n <= n_available:
        run(n)
