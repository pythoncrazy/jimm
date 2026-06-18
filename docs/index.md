# jimm docs

jimm is a JAX image-model library built on Flax NNX. It supports loading pretrained weights from HuggingFace, hardware-accelerated attention, and FSDP-style explicit sharding.

## Models Implemented

* Vision Transformers (CLS pooling or Multihead Attention Pooling)
* CLIP
* SigLIP
* DINOv2
* DINOv3
* AIMv2
* Many more on the way

## Flash / Splash Attention (via Tokamax)

All models accept an `attention_fn` argument for hardware-accelerated attention using [Tokamax](https://github.com/openxla/tokamax):

| Backend | Hardware | Notes |
|---------|----------|-------|
| `"mosaic"` | NVIDIA H100 (SM90) / B100 (SM100) | Pallas Mosaic GPU kernel |
| `"triton"` | Any NVIDIA GPU | Pallas Triton kernel |
| `"cudnn"` | NVIDIA GPU | Via JAX-NN / cuDNN |
| `"mosaic_tpu"` | TPU v5+ (all generations) | Splash attention (block-sparse) |
| `"xla_chunked"` | GPU / TPU | Flash-style chunked XLA |
| `"xla"` | Any | Standard XLA fallback |

Pass a list for automatic fallback:

```python
import jimm

# GPU: try H100 Mosaic kernel, fall back to Triton, then XLA
model = jimm.CLIP.from_pretrained("openai/clip-vit-large-patch14",
                                   attention_fn=jimm.make_tokamax_attention(["mosaic", "triton", "xla"]))

# TPU: try Splash attention, fall back to chunked XLA
model = jimm.CLIP.from_pretrained("openai/clip-vit-large-patch14",
                                   attention_fn=jimm.make_tokamax_attention(["mosaic_tpu", "xla_chunked"]))
```

> **Note:** Flash/Splash attention does not provide a speedup at typical vision/text context lengths (e.g. 256 image tokens, 77 text tokens). The primary benefit is memory reduction at longer sequence lengths.

## FSDP / Explicit Sharding

All models support JAX explicit sharding (FSDP-style) out of the box. Set up a mesh before model creation:

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

model = jimm.CLIP.from_pretrained("openai/clip-vit-large-patch14")
# params are automatically sharded across fsdp axis
```

Each model ships with a default sharding config (`CLIPSharding`, `SigLIPSharding`, `ViTSharding`) that shards large weight matrices on the contracting (`in_features`) dimension, keeping activations batch-sharded only. Specs are per-layer shapes; the Transformer stack patches Variable metadata after `nnx.vmap` so the optimizer (e.g. AdamW via `nnx.Optimizer`) initialises its state with the correct stacked spec — no manual fixups needed.

Pass `sharding=jimm.common.sharding.NoSharding()` to disable all sharding.

## Installation
### Using pixi.sh:
`pixi add jimm@https://github.com/pythoncrazy/jimm.git --pypi`
### Using uv
`uv add git+https://github.com/pythoncrazy/jimm.git`
or if you prefer to not add as a direct dependency:
`uv pip install git+https://github.com/pythoncrazy/jimm.git`
### Using pip/conda
`pip install git+https://github.com/pythoncrazy/jimm.git`
