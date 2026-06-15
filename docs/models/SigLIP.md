# SigLIP

SigLIP (Sigmoid Loss for Language-Image Pre-Training) is a vision-language model that replaces CLIP's softmax contrastive loss with a pairwise sigmoid loss. This treats each image-text pair as an independent binary classification problem, removing the need for global batch normalization and enabling efficient training on larger batches.

Paper: ["Sigmoid Loss for Language Image Pre-Training"](https://arxiv.org/abs/2303.15343) (Zhai et al., ICCV 2023)
Code: [github.com/google-research/big_vision](https://github.com/google-research/big_vision)

SigLIP uses a Vision Transformer with a Multi-Head Attention Pooling (MAP) head as the image encoder, and a standard Transformer as the text encoder. The jimm implementation supports the full model (`SigLIP`), the vision encoder alone (`SigLIPVisionModel`), and the text encoder alone (`SigLIPTextModel`).

## Supported models

| HuggingFace ID | Vision arch | `hidden_size` | Image size |
|---|---|---|---|
| `google/siglip-base-patch16-224` | ViT-B/16 | 768 | 224 |
| `google/siglip-base-patch16-256` | ViT-B/16 | 768 | 256 |
| `google/siglip-large-patch16-384` | ViT-L/16 | 1024 | 384 |
| `google/siglip-so400m-patch14-384` | SoViT-400M/14 | 1152 | 384 |

## Basic usage

```python
import jimm
import numpy as np

model = jimm.SigLIP.from_pretrained("google/siglip-base-patch16-256")

images = np.random.rand(4, 256, 256, 3).astype(np.float32)
text = np.ones((4, 64), dtype=np.int32)  # tokenized text, shape (batch, seq_len)
logits = model(images, text)  # shape: (4, 4)
```

## Flash / Splash Attention

SigLIP supports hardware-accelerated attention via [Tokamax](https://github.com/openxla/tokamax). Pass an `attention_fn` at construction time:

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
model = jimm.SigLIP.from_pretrained("google/siglip-base-patch16-256",
                                     attention_fn=jimm.make_tokamax_attention(["mosaic", "triton", "xla"]))

# TPU: try Splash attention, fall back to chunked XLA
model = jimm.SigLIP.from_pretrained("google/siglip-base-patch16-256",
                                     attention_fn=jimm.make_tokamax_attention(["mosaic_tpu", "xla_chunked"]))
```

> **Note:** Flash/Splash attention does not provide a speedup at typical SigLIP context lengths (256 image tokens, 64 text tokens). The primary benefit is memory reduction at longer sequence lengths.

## FSDP / Explicit Sharding

SigLIP supports JAX explicit sharding (FSDP-style) via `SigLIPSharding`. Large weight matrices are sharded on the contracting (`in_features`) dimension so that activations carry only the batch-axis sharding.

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

model = jimm.SigLIP.from_pretrained("google/siglip-base-patch16-256")
# model params are automatically sharded across fsdp axis
```

`SigLIPSharding` specs represent **per-layer** shapes. The `Transformer` stack prepends `None` for the scan axis to Variable metadata after `nnx.vmap`, so the optimizer receives the correct stacked spec natively.

To disable sharding, pass `sharding=jimm.common.sharding.NoSharding()`.

::: jimm.models.siglip.SigLIPVisionModel
    options:
        show_root_heading: true
        show_source: true


::: jimm.models.siglip.SigLIPTextModel
    options:
        show_root_heading: true
        show_source: true


::: jimm.models.siglip.SigLIP
    options:
        show_root_heading: true
        show_source: true
