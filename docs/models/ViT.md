# ViT

ViT (Vision Transformer) applies a standard Transformer encoder directly to sequences of image patches. Each image is split into fixed-size patches, linearly projected into a hidden dimension, prepended with a learnable CLS token, and processed with learned absolute position embeddings.

Paper: ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929) (Dosovitskiy et al., ICLR 2021)
Code: [github.com/google-research/vision_transformer](https://github.com/google-research/vision_transformer)

The jimm implementation returns the CLS token embedding after the final LayerNorm.

## Supported models

| HuggingFace ID | `hidden_size` | Layers | Heads | Patch | Image size |
|---|---|---|---|---|---|
| `google/vit-base-patch16-224` | 768 | 12 | 12 | 16 | 224 |
| `google/vit-base-patch32-224-in21k` | 768 | 12 | 12 | 32 | 224 |
| `google/vit-large-patch16-224` | 1024 | 24 | 16 | 16 | 224 |

## Basic usage

```python
import jimm
import numpy as np

model = jimm.VisionTransformer.from_pretrained("google/vit-base-patch16-224")

images = np.random.rand(4, 224, 224, 3).astype(np.float32)
embeddings = model(images)  # shape: (4, 768)
```

## Flash / Splash Attention

ViT supports hardware-accelerated attention via [Tokamax](https://github.com/openxla/tokamax):

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
model = jimm.VisionTransformer.from_pretrained("google/vit-base-patch16-224",
                                                attention_fn=jimm.make_tokamax_attention(["mosaic", "triton", "xla"]))

# TPU: try Splash attention, fall back to chunked XLA
model = jimm.VisionTransformer.from_pretrained("google/vit-base-patch16-224",
                                                attention_fn=jimm.make_tokamax_attention(["mosaic_tpu", "xla_chunked"]))
```

> **Note:** Flash/Splash attention does not provide a speedup at typical ViT context lengths (196 tokens for 224px/16px). The primary benefit is memory reduction at longer sequence lengths.

## FSDP / Explicit Sharding

ViT supports JAX explicit sharding (FSDP-style) via `ViTSharding`. Large weight matrices are sharded on the contracting (`in_features`) dimension so that activations carry only the batch-axis sharding.

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

model = jimm.VisionTransformer.from_pretrained("google/vit-base-patch16-224")
# model params are automatically sharded across fsdp axis
```

`ViTSharding` specs represent **per-layer** shapes. The `Transformer` stack prepends `None` for the scan axis to Variable metadata after `nnx.vmap`, so the optimizer receives the correct stacked spec natively.

To disable sharding, pass `sharding=jimm.common.sharding.NoSharding()`.

::: jimm.models.vit.VisionTransformer
    options:
        show_root_heading: true
        show_source: true
