# AIMv2

AIMv2 is a vision transformer trained autoregressively on image-text pairs. Rather than a contrastive or reconstruction objective, it predicts image tokens in a causal left-to-right sequence conditioned on a multimodal prefix, producing strong general-purpose patch-level representations.

Paper: ["Multimodal Autoregressive Pre-training of Large Vision Encoders"](https://arxiv.org/abs/2411.14402) (El-Nouby et al., CVPR 2025)
Code: [github.com/apple/ml-aim](https://github.com/apple/ml-aim/tree/main/aim-v2)

The jimm implementation returns **all patch token embeddings** after the final RMSNorm with shape `(batch, n_patches, hidden_size)`. AIMv2 has no CLS token; downstream tasks pool over patches or select specific tokens.

Key architectural differences from standard ViT:

| Feature | ViT | AIMv2 |
|---|---|---|
| Normalization | LayerNorm | RMSNorm |
| MLP | Standard GELU | SwiGLU |
| Pre-norm placement | After patch + pos | On patches only, before pos |
| CLS token | ✅ | ❌ |
| Output | CLS embedding `(B, D)` | All patches `(B, N, D)` |
| Attention / MLP biases | Yes | No |

## Supported models

All AIMv2 models use patch size 14 and 24 transformer layers regardless of scale. The four architecture sizes scale width only.

### Large - 307M params

| HuggingFace ID | `hidden_size` | Heads | `intermediate_size` | Image size |
|---|---|---|---|---|
| `apple/aimv2-large-patch14-224` | 1024 | 8 | 2816 | 224 |
| `apple/aimv2-large-patch14-336` | 1024 | 8 | 2816 | 336 |
| `apple/aimv2-large-patch14-448` | 1024 | 8 | 2816 | 448 |
| `apple/aimv2-large-patch14-224-distilled` | 1024 | 8 | 2816 | 224 |
| `apple/aimv2-large-patch14-224-lit` | 1024 | 8 | 2816 | 224 |

### Huge - 680M params

| HuggingFace ID | `hidden_size` | Heads | `intermediate_size` | Image size |
|---|---|---|---|---|
| `apple/aimv2-huge-patch14-224` | 1536 | 12 | 4096 | 224 |
| `apple/aimv2-huge-patch14-336` | 1536 | 12 | 4096 | 336 |
| `apple/aimv2-huge-patch14-448` | 1536 | 12 | 4096 | 448 |

### 1B - 1.2B params

| HuggingFace ID | `hidden_size` | Heads | `intermediate_size` | Image size |
|---|---|---|---|---|
| `apple/aimv2-1B-patch14-224` | 2048 | 16 | 5632 | 224 |
| `apple/aimv2-1B-patch14-336` | 2048 | 16 | 5632 | 336 |
| `apple/aimv2-1B-patch14-448` | 2048 | 16 | 5632 | 448 |

### 3B - 2.7B params

| HuggingFace ID | `hidden_size` | Heads | `intermediate_size` | Image size |
|---|---|---|---|---|
| `apple/aimv2-3B-patch14-224` | 3072 | 24 | 8192 | 224 |
| `apple/aimv2-3B-patch14-336` | 3072 | 24 | 8192 | 336 |
| `apple/aimv2-3B-patch14-448` | 3072 | 24 | 8192 | 448 |

> **Note:** `apple/aimv2-large-patch14-native` uses sinusoidal position embeddings for variable resolution and is not currently supported. The lit model is a full multimodal checkpoint (vision + text encoder + projectors); `AIMv2Model.from_pretrained` automatically extracts just the vision encoder from it.

## Basic usage

```python
import jimm
import numpy as np

model = jimm.AIMv2Model.from_pretrained("apple/aimv2-large-patch14-224")

images = np.random.rand(4, 224, 224, 3).astype(np.float32)
patch_embeddings = model(images)  # shape: (4, 256, 1024)
```

The model returns all 256 patch tokens (16×16 grid) for a 224×224 image with patch size 14. To get a single image embedding, pool over patches:

```python
# Mean pool across patch dimension
image_embeddings = patch_embeddings.mean(axis=1)  # shape: (4, 1024)
```

## Flash / Splash Attention

AIMv2 supports hardware-accelerated attention via [Tokamax](https://github.com/openxla/tokamax):

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
model = jimm.AIMv2Model.from_pretrained("apple/aimv2-large-patch14-448",
                                         attention_fn=jimm.make_tokamax_attention(["mosaic", "triton", "xla"]))

# TPU: try Splash attention, fall back to chunked XLA
model = jimm.AIMv2Model.from_pretrained("apple/aimv2-large-patch14-448",
                                         attention_fn=jimm.make_tokamax_attention(["mosaic_tpu", "xla_chunked"]))
```

> **Note:** AIMv2-Large produces 256 patch tokens at 224×224, 576 at 336×336, and 1024 at 448×448. Flash attention is most beneficial at the larger image sizes.

## FSDP / Explicit Sharding

AIMv2 supports JAX explicit sharding (FSDP-style) via `AIMv2Sharding`, which inherits `ViTSharding` with no overrides needed (AIMv2 has no LayerScale, no register tokens, and no MLP bias tensors to shard).

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

model = jimm.AIMv2Model.from_pretrained("apple/aimv2-large-patch14-224")
# model params are automatically sharded across fsdp axis
```

To disable sharding, pass `sharding=NoSharding()` (import: `from jimm.common.sharding import NoSharding`).

::: jimm.models.aimv2.AIMv2Model
    options:
        show_root_heading: true
        show_source: true
