# DINOv3

DINOv3 is a self-supervised vision transformer that extends DINOv2 with three architectural changes designed to improve scalability and training stability: **2D Rotary Position Embeddings (RoPE)**, **register tokens**, and an optional **gated MLP (SwiGLU)**. It is trained via self-distillation on a large curated dataset, producing general-purpose image representations that transfer well to downstream tasks.

The jimm implementation returns the CLS token embedding after the final LayerNorm. Because RoPE computes position embeddings dynamically from the input spatial dimensions, the model accepts **variable image sizes** — any height and width divisible by `patch_size` — without retraining or interpolation.

Key architectural differences from DINOv2:

| Feature | DINOv2 | DINOv3 |
|---|---|---|
| Position embeddings | Learned absolute (fixed grid) | 2D RoPE (dynamic, variable size) |
| Register tokens | None | 4 learnable tokens between CLS and patches |
| MLP | Standard (fc1 → GELU → fc2) | Standard or gated (SwiGLU) |
| key bias | Yes | No |

## Supported models

| HuggingFace ID | `hidden_size` | Layers | Heads | Patch | Image size |
|---|---|---|---|---|---|
| `facebook/dinov3-vits16-pretrain-lvd1689m` | 384 | 12 | 6 | 16 | variable |
| `facebook/dinov3-vitb16-pretrain-lvd1689m` | 768 | 12 | 12 | 16 | variable |

> **Note:** These are gated-repo models — you must request access on HuggingFace and authenticate with `huggingface-cli login` (or set `HF_TOKEN`) before loading.

## Basic usage

```python
import jimm
import numpy as np

model = jimm.DINOv3Model.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")

# Standard 224×224 input
images = np.random.rand(4, 224, 224, 3).astype(np.float32)
embeddings = model(images)  # shape: (4, 384)

# Variable image size — any multiple of patch_size (16)
images_rect = np.random.rand(4, 192, 256, 3).astype(np.float32)
embeddings_rect = model(images_rect)  # shape: (4, 384)
```

## Variable image sizes

Unlike DINOv2's fixed learned position embedding table, DINOv3's RoPE is recomputed on every forward pass from the actual spatial dimensions. This means you can pass images of any size divisible by `patch_size` without any interpolation artefacts:

```python
import jimm

model = jimm.DINOv3Model.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")

# All valid — RoPE adapts to each resolution
model(images_224)      # 224×224 → 196 patch tokens
model(images_336)      # 336×336 → 441 patch tokens
model(images_192x256)  # 192×256 → 192 patch tokens
```

> **Note:** JAX traces the model once per unique `(height, width)` pair. If you benchmark across many resolutions the first call at each size will be slower due to recompilation. For production use, fix the image size or pre-warm the shapes you need.

## Flash / Splash Attention

DINOv3 supports hardware-accelerated attention via [Tokamax](https://github.com/openxla/tokamax):

| Backend | Hardware | Notes |
|---------|----------|-------|
| `"mosaic"` | NVIDIA H100 (SM90) / B100 (SM100) | Pallas Mosaic GPU kernel |
| `"triton"` | Any NVIDIA GPU | Pallas Triton kernel |
| `"cudnn"` | NVIDIA GPU | Via JAX-NN / cuDNN |
| `"mosaic_tpu"` | TPU v5+ (all generations) | Splash attention (block-sparse) |
| `"xla_chunked"` | GPU / TPU | Flash-style chunked XLA |
| `"xla"` | Any | Standard XLA fallback |

```python
import jimm

# GPU: try H100 Mosaic kernel, fall back to Triton, then XLA
model = jimm.DINOv3Model.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m",
                                          attention_fn=jimm.make_tokamax_attention(["mosaic", "triton", "xla"]))

# TPU: try Splash attention, fall back to chunked XLA
model = jimm.DINOv3Model.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m",
                                          attention_fn=jimm.make_tokamax_attention(["mosaic_tpu", "xla_chunked"]))
```

> **Note:** The RoPE attention path bypasses the standard `nnx.MultiHeadAttention.__call__` and applies rotations manually, so the custom `attention_fn` is not used in the RoPE code path. Flash attention is applied in the standard (non-RoPE) path only; the RoPE path always uses XLA dot-product attention.

## FSDP / Explicit Sharding

DINOv3 supports JAX explicit sharding (FSDP-style) via `DINOv3Sharding`, which extends `ViTSharding` with sharding for LayerScale vectors.

```python
from jax.experimental import mesh_utils
from jax.sharding import AxisType, Mesh
import jax

n_devices = jax.device_count()
mesh = Mesh(
    mesh_utils.create_device_mesh((1, n_devices)),
    ("data", "fsdp"),
    axis_types=(AxisType.Explicit, AxisType.Explicit),
)
jax.set_mesh(mesh)

model = jimm.DINOv3Model.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
# model params are automatically sharded across fsdp axis
```

To disable sharding, pass `sharding=NoSharding()` (import: `from jimm.common.sharding import NoSharding`).

::: jimm.models.dinov3.DINOv3Model
    options:
        show_root_heading: true
        show_source: true
