# ViT (Vision Transformer)

The ViT (Vision Transformer) is a transformer-based neural network architecture for image classification. It divides an image into fixed-size patches, linearly embeds each patch, adds position embeddings, and processes the resulting sequence of vectors through a standard transformer encoder.

The ViT model was introduced in the paper ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929) and has shown strong performance on image classification benchmarks.

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

> **Note:** Flash/Splash attention does not provide a speedup at typical ViT context lengths (e.g. 196 tokens for 224px/16px). The primary benefit is memory reduction at longer sequence lengths.

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
```

`ViTSharding` specs represent **per-layer** shapes. The `Transformer` stack prepends `None` for the scan axis to Variable metadata after `nnx.vmap`, so the optimizer receives the correct stacked spec natively.

To disable sharding, pass `sharding=jimm.common.sharding.NoSharding()`.

::: jimm.models.vit.VisionTransformer
    options:
        show_root_heading: true
        show_source: true