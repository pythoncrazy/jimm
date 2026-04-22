"""Benchmark tokamax.linear_softmax_cross_entropy_loss vs standard for CLIP at BS=8192."""

import time

import jax
import jax.numpy as jnp
import optax
import tokamax
from jax.sharding import Mesh

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

BATCH_SIZE = 8192
EMBED_DIM = 768  # StreetCLIP / ViT-L/14
N_WARMUP = 5
N_BENCH = 20

mesh = Mesh(jax.devices(), ("fsdp",))
jax.set_mesh(mesh)
print(f"Devices: {jax.devices()}")
print(f"Batch size: {BATCH_SIZE}, Embed dim: {EMBED_DIM}")


def standard_clip_loss(image_features, text_features, logit_scale):
    image_features = image_features / jnp.linalg.norm(image_features, axis=-1, keepdims=True)
    text_features = text_features / jnp.linalg.norm(text_features, axis=-1, keepdims=True)
    logits = jnp.exp(logit_scale) * image_features @ text_features.T
    labels = jnp.arange(image_features.shape[0])
    image_loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    text_loss = optax.softmax_cross_entropy_with_integer_labels(logits.T, labels).mean()
    return (image_loss + text_loss) / 2.0


def tokamax_clip_loss(image_features, text_features, logit_scale):
    image_features = image_features / jnp.linalg.norm(image_features, axis=-1, keepdims=True)
    text_features = text_features / jnp.linalg.norm(text_features, axis=-1, keepdims=True)
    scale = jnp.exp(logit_scale)
    labels = jnp.arange(image_features.shape[0])
    # Scale x so logits = (scale*x) @ w = scale * (x @ w)
    image_loss = tokamax.linear_softmax_cross_entropy_loss(scale * image_features, labels, text_features.T, reduction="mean")
    text_loss = tokamax.linear_softmax_cross_entropy_loss(scale * text_features, labels, image_features.T, reduction="mean")
    return (image_loss + text_loss) / 2.0


def make_inputs(key=0):
    rng = jax.random.PRNGKey(key)
    k1, k2 = jax.random.split(rng)
    img = jax.random.normal(k1, (BATCH_SIZE, EMBED_DIM), dtype=jnp.bfloat16)
    txt = jax.random.normal(k2, (BATCH_SIZE, EMBED_DIM), dtype=jnp.bfloat16)
    scale = jnp.array(2.0)
    return img, txt, scale


def bench(fn, img, txt, scale, label):
    grad_fn = jax.value_and_grad(fn, argnums=(0, 1, 2))
    jit_fn = jax.jit(grad_fn)

    # warmup
    for _ in range(N_WARMUP):
        loss, grads = jit_fn(img, txt, scale)
        loss.block_until_ready()

    times = []
    for _ in range(N_BENCH):
        t0 = time.perf_counter()
        loss, grads = jit_fn(img, txt, scale)
        loss.block_until_ready()
        times.append(time.perf_counter() - t0)

    times_ms = [t * 1000 for t in times]
    mean_ms = sum(times_ms) / len(times_ms)
    min_ms = min(times_ms)
    print(f"{label}: mean={mean_ms:.2f}ms  min={min_ms:.2f}ms  loss={float(loss):.4f}")
    return mean_ms, min_ms, float(loss)


def main():
    img, txt, scale = make_inputs()

    print("\n--- Autotuning tokamax CLIP loss ---")
    result = tokamax.autotune(tokamax_clip_loss, img, txt, scale)
    print(f"Autotuning complete: {result}")

    print("\n--- Benchmarking (forward + backward) ---")
    with result:
        std_mean, std_min, std_loss = bench(standard_clip_loss, img, txt, scale, "standard  ")
        tok_mean, tok_min, tok_loss = bench(tokamax_clip_loss, img, txt, scale, "tokamax   ")

    print(f"\nLoss match: {abs(std_loss - tok_loss) < 0.01}")
    print(f"Speedup (mean): {std_mean / tok_mean:.2f}x")
    print(f"Speedup (min):  {std_min / tok_min:.2f}x")

    print("\n--- Memory: logit matrix size at BS=8192 ---")
    logit_bytes = BATCH_SIZE * BATCH_SIZE * 2  # bfloat16
    print(f"Standard materializes {logit_bytes / 1e6:.1f}MB logit matrix per direction")
    print("Tokamax mosaic_tpu avoids this entirely")


if __name__ == "__main__":
    main()
