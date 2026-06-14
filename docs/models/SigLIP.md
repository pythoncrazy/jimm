# SigLIP (Sigmoid Loss for Language Image Pre-Training)

SigLIP (Sigmoid Loss for Language Image Pre-Training) is a vision-language model that builds upon the principles of CLIP but introduces a key architectural change: it uses a sigmoid loss function instead of the softmax-based contrastive loss. Additionally, there are some slight implementation differences (no attention_mask for the text encoder, padding the text inputs, multihead attention pooling for the vision encoder rather than a linear projection layer).

This modification simplifies the training objective by treating the problem as a binary classification for each image-text pair (i.e., are they a positive or negative match?). This approach avoids the need for a global normalization over all pairs in a batch, which makes it more scalable and robust to noisy, web-scale data.

Key features of SigLIP:
1.  **Vision Encoder**: A Vision Transformer (ViT) with a Multi-Head Attention Pooling (MAP) head.
2.  **Text Encoder**: A standard Transformer model.
3.  **Sigmoid Loss**: Enables training on larger batches and noisier datasets without requiring careful data curation or complex negative sampling strategies.

SigLIP was introduced in the paper ["Sigmoid Loss for Language Image Pre-Training"](https://arxiv.org/abs/2303.15343) and has demonstrated improved performance and training efficiency.

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

SigLIP supports JAX explicit sharding (FSDP-style) out of the box via `SigLIPSharding`. Large weight matrices are sharded on the contracting (`in_features`) dimension so that activations carry only the batch-axis sharding.

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
```

`SigLIPSharding` specs represent **per-layer** shapes. The `Transformer` stack prepends `None` for the scan axis to the Variable metadata after `nnx.vmap`, so the optimizer receives the correct stacked spec natively without any manual fixups.

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
