# DINOv2

DINOv2 is a self-supervised vision transformer trained on a large curated image dataset without manual annotation. It uses a self-distillation objective and produces general-purpose image representations that transfer well across downstream tasks.

Paper: ["DINOv2: Learning Robust Visual Features without Supervision"](https://arxiv.org/abs/2304.07193) (Oquab et al., 2023)
Code: [github.com/facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)

The jimm implementation returns the CLS token embedding after the final LayerNorm. The key architectural difference from standard ViT is **LayerScale**: each transformer block multiplies the attention and MLP residuals by a learnable per-channel scalar vector before the residual add, stabilizing training at scale.

## Supported models

| HuggingFace ID | `hidden_size` | Layers | Heads | Patch | Image size |
|---|---|---|---|---|---|
| `facebook/dinov2-small` | 384 | 12 | 6 | 14 | 518 |
| `facebook/dinov2-base` | 768 | 12 | 12 | 14 | 518 |
| `facebook/dinov2-large` | 1024 | 24 | 16 | 14 | 518 |
| `facebook/dinov2-giant` | 1536 | 40 | 24 | 14 | 518 |

## Basic usage

```python
import jimm
import numpy as np

model = jimm.DINOv2Model.from_pretrained("facebook/dinov2-small")

images = np.random.rand(4, 518, 518, 3).astype(np.float32)
embeddings = model(images)  # shape: (4, 384)
```

## Flash / Splash Attention

DINOv2 supports hardware-accelerated attention via [Tokamax](https://github.com/openxla/tokamax):

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
model = jimm.DINOv2Model.from_pretrained("facebook/dinov2-base",
                                          attention_fn=jimm.make_tokamax_attention(["mosaic", "triton", "xla"]))

# TPU: try Splash attention, fall back to chunked XLA
model = jimm.DINOv2Model.from_pretrained("facebook/dinov2-base",
                                          attention_fn=jimm.make_tokamax_attention(["mosaic_tpu", "xla_chunked"]))
```

> **Note:** DINOv2 processes 1370 tokens at 518x518 with patch size 14, which is substantially longer than typical ViT sequences. Flash attention provides a meaningful memory reduction at this sequence length.

## FSDP / Explicit Sharding

DINOv2 supports JAX explicit sharding (FSDP-style) via `DINOv2Sharding`, which extends `ViTSharding` with additional sharding for the LayerScale vectors.

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

model = jimm.DINOv2Model.from_pretrained("facebook/dinov2-base")
# model params are automatically sharded across fsdp axis
```

`DINOv2Sharding` specs represent **per-layer** shapes. The `Transformer` stack prepends `None` for the scan axis to Variable metadata after `nnx.vmap`, so the optimizer receives the correct stacked spec natively.

To disable sharding, pass `sharding=NoSharding()` (import: `from jimm.common.sharding import NoSharding`).

::: jimm.models.dinov2.DINOv2Model
    options:
        show_root_heading: true
        show_source: true
