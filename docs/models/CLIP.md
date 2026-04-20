# CLIP (Contrastive Language–Image Pre-training)

CLIP (Contrastive Language–Image Pre-training) is a neural network architecture that learns visual concepts from natural language supervision. It is trained on a large dataset of image-text pairs to create a unified vision-language model that can understand both images and text in a shared semantic space.

CLIP consists of two main components:
1. A vision encoder (Vision Transformer) that processes images into visual features
2. A text encoder (Transformer) that processes text into textual features

The model is trained using contrastive learning, where it learns to maximize the cosine similarity between the embeddings of matching image-text pairs while minimizing it for non-matching pairs. This allows CLIP to perform zero-shot classification by comparing image embeddings with text embeddings of potential labels.

CLIP was introduced in the paper ["Learning Transferable Visual Models From Natural Language Supervision"](https://arxiv.org/abs/2103.00020) and has shown remarkable zero-shot generalization capabilities across a wide range of visual classification tasks. The CLIP model combines a Vision Transformer and a Text Transformer to learn joint representations of images and text. It is trained to maximize the similarity between matching image-text pairs while minimizing similarity between non-matching pairs.

## Flash / Splash Attention

CLIP supports hardware-accelerated attention via [Tokamax](https://github.com/openxla/tokamax). Pass an `attention_fn` at construction time:

| Backend | Hardware | Notes |
|---------|----------|-------|
| `"mosaic"` | NVIDIA H100 (SM90) / B100 (SM100) | Pallas Mosaic GPU kernel |
| `"triton"` | Any NVIDIA GPU | Pallas Triton kernel |
| `"cudnn"` | NVIDIA GPU | Via JAX-NN / cuDNN |
| `"mosaic_tpu"` | TPU v5 / v7 | Splash attention (block-sparse) |
| `"xla_chunked"` | GPU / TPU | Flash-style chunked XLA |
| `"xla"` | Any | Standard XLA fallback |

```python
import jimm

# GPU: try H100 Mosaic kernel, fall back to Triton, then XLA
model = jimm.CLIP.from_pretrained("openai/clip-vit-large-patch14",
                                   attention_fn=jimm.make_tokamax_attention(["mosaic", "triton", "xla"]))

# TPU: try Splash attention, fall back to chunked XLA
model = jimm.CLIP.from_pretrained("openai/clip-vit-large-patch14",
                                   attention_fn=jimm.make_tokamax_attention(["mosaic_tpu", "xla_chunked"]))
```

You can also apply different kernels to each encoder via `vision_attention_fn` and `text_attention_fn`.

> **Note:** Flash/Splash attention does not provide a speedup at typical CLIP context lengths (256 image tokens, 77 text tokens). The primary benefit is memory reduction at longer sequence lengths.

## FSDP / Explicit Sharding

CLIP supports JAX explicit sharding (FSDP-style) out of the box via `CLIPSharding`. Large weight matrices are sharded on the contracting (`in_features`) dimension so that activations carry only the batch-axis sharding, avoiding duplicate-axis conflicts.

```python
from jax.experimental import mesh_utils
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
import jax

n_devices = jax.device_count()
mesh = Mesh(
    mesh_utils.create_device_mesh((1, n_devices)),
    ("data", "fsdp"),
    axis_types=(AxisType.Explicit, AxisType.Explicit),
)
jax.set_mesh(mesh)

model = jimm.CLIP.from_pretrained("openai/clip-vit-large-patch14")
# model params are automatically sharded across fsdp axis
```

`CLIPSharding` specs represent **per-layer** shapes. The `Transformer` stack prepends `None` for the scan axis to the Variable metadata after `nnx.vmap`, so the optimizer (e.g. `nnx.Optimizer` with AdamW) receives the correct stacked spec and initialises its state without any manual fixups.

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
