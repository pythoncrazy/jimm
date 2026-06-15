# CLIP

CLIP (Contrastive Language-Image Pre-Training) is a vision-language model trained on 400 million image-text pairs using a contrastive objective. It learns a shared embedding space where matching image-text pairs have high cosine similarity and non-matching pairs have low similarity, enabling zero-shot classification and cross-modal retrieval.

Paper: ["Learning Transferable Visual Models From Natural Language Supervision"](https://arxiv.org/abs/2103.00020) (Radford et al., 2021)
Code: [github.com/openai/CLIP](https://github.com/openai/CLIP)

CLIP consists of a Vision Transformer (ViT) image encoder and a causal Transformer text encoder. Both produce fixed-size embeddings that are compared via a temperature-scaled dot product. The jimm implementation supports the full model (`CLIP`), the vision encoder alone (`CLIPVisionModel`), and the text encoder alone (`CLIPTextModel`).

## Supported models

| HuggingFace ID | Vision arch | Text `hidden_size` | Image size |
|---|---|---|---|
| `openai/clip-vit-base-patch32` | ViT-B/32 | 512 | 224 |
| `openai/clip-vit-base-patch16` | ViT-B/16 | 512 | 224 |
| `openai/clip-vit-large-patch14` | ViT-L/14 | 768 | 224 |
| `openai/clip-vit-large-patch14-336` | ViT-L/14 | 768 | 336 |

> **Note:** OpenAI CLIP weights are distributed as `pytorch_model.bin`. Pass `use_pytorch=True` when loading.

## Basic usage

```python
import jimm
import numpy as np

model = jimm.CLIP.from_pretrained("openai/clip-vit-large-patch14", use_pytorch=True)

images = np.random.rand(4, 224, 224, 3).astype(np.float32)
text = np.array([[49406, 1234, 49407, 0, 0]])  # tokenized text, shape (batch, seq_len)
logits = model(images, text)  # shape: (4, 1)
```

## Flash / Splash Attention

CLIP supports hardware-accelerated attention via [Tokamax](https://github.com/openxla/tokamax). Pass an `attention_fn` at construction time:

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
model = jimm.CLIP.from_pretrained("openai/clip-vit-large-patch14", use_pytorch=True,
                                   attention_fn=jimm.make_tokamax_attention(["mosaic", "triton", "xla"]))

# TPU: try Splash attention, fall back to chunked XLA
model = jimm.CLIP.from_pretrained("openai/clip-vit-large-patch14", use_pytorch=True,
                                   attention_fn=jimm.make_tokamax_attention(["mosaic_tpu", "xla_chunked"]))
```

You can also apply different kernels to each encoder via `vision_attention_fn` and `text_attention_fn`.

> **Note:** Flash/Splash attention does not provide a speedup at typical CLIP context lengths (256 image tokens, 77 text tokens). The primary benefit is memory reduction at longer sequence lengths.

## FSDP / Explicit Sharding

CLIP supports JAX explicit sharding (FSDP-style) via `CLIPSharding`. Large weight matrices are sharded on the contracting (`in_features`) dimension so that activations carry only the batch-axis sharding.

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

model = jimm.CLIP.from_pretrained("openai/clip-vit-large-patch14", use_pytorch=True)
# model params are automatically sharded across fsdp axis
```

`CLIPSharding` specs represent **per-layer** shapes. The `Transformer` stack prepends `None` for the scan axis to Variable metadata after `nnx.vmap`, so the optimizer receives the correct stacked spec natively.

To disable sharding, pass `sharding=jimm.common.sharding.NoSharding()`.

::: jimm.models.clip.CLIPVisionModel
    options:
        show_root_heading: true
        show_source: true


::: jimm.models.clip.CLIPTextModel
    options:
        show_root_heading: true
        show_source: true


::: jimm.models.clip.CLIP
    options:
        show_root_heading: true
        show_source: true
