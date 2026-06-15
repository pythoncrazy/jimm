# DINOv3

DINOv3 is a self-supervised vision transformer that extends the DINOv2 training recipe with three architectural improvements: 2D Rotary Position Embeddings (RoPE), register tokens, and a gated MLP (SwiGLU). It is trained on LVD-1689M, a curated dataset of 1.69 billion images, and produces general-purpose image representations that transfer well across downstream tasks.

Paper: ["DINOv3"](https://arxiv.org/abs/2508.10104) (Siméoni et al., 2025)
Code: [github.com/facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)

The jimm implementation returns the CLS token embedding after the final LayerNorm. Because RoPE computes position embeddings dynamically from the input spatial dimensions, the model accepts **variable image sizes**: any height and width divisible by `patch_size`, without retraining or interpolation.

Key architectural differences from DINOv2:

| Feature | DINOv2 | DINOv3 |
|---|---|---|
| Position embeddings | Learned absolute (fixed grid) | 2D RoPE (dynamic, variable size) |
| Register tokens | None | 4 learnable tokens between CLS and patches |
| MLP | Standard (fc1 -> GELU -> fc2) | Gated SwiGLU |
| Key bias | Yes | No |

Register tokens were introduced in ["Vision Transformers Need Registers"](https://arxiv.org/abs/2309.16588) (Darcet et al., 2024).

## Supported models

| HuggingFace ID | `hidden_size` | Layers | Heads | Patch | Image size |
|---|---|---|---|---|---|
| `facebook/dinov3-vits16-pretrain-lvd1689m` | 384 | 12 | 6 | 16 | variable |
| `facebook/dinov3-vitb16-pretrain-lvd1689m` | 768 | 12 | 12 | 16 | variable |
| `facebook/dinov3-vitl16-pretrain-lvd1689m` | 1024 | 24 | 16 | 16 | variable |

> **Note:** These are gated-repo models. You must request access on HuggingFace and authenticate with `huggingface-cli login` (or set `HF_TOKEN`) before loading.

## Basic usage

```python
import jimm
import numpy as np

model = jimm.DINOv3Model.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")

# Standard 224x224 input
images = np.random.rand(4, 224, 224, 3).astype(np.float32)
embeddings = model(images)  # shape: (4, 384)

# Variable image size -- any multiple of patch_size (16)
images_rect = np.random.rand(4, 192, 256, 3).astype(np.float32)
embeddings_rect = model(images_rect)  # shape: (4, 384)
```

## Variable image sizes

Unlike DINOv2's fixed learned position embedding table, DINOv3's RoPE is recomputed on every forward pass from the actual spatial dimensions. Any height and width divisible by `patch_size` work without interpolation:

```python
import jimm

model = jimm.DINOv3Model.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")

model(images_224)      # 224x224 -> 196 patch tokens
model(images_336)      # 336x336 -> 441 patch tokens
model(images_192x256)  # 192x256 -> 192 patch tokens
```

> **Note:** JAX traces the model once per unique `(height, width)` pair. For production use, fix the image size or pre-warm the shapes you need.

## Flash / Splash Attention

DINOv3 does not support custom attention functions. The RoPE path applies Q/K rotations and scaled dot-product attention directly in JAX, bypassing `nnx.MultiHeadAttention.__call__`. Passing `attention_fn` raises `ValueError`.

## FSDP / Explicit Sharding

DINOv3 supports JAX explicit sharding (FSDP-style) via `DINOv3Sharding`, which extends `ViTSharding` with sharding for LayerScale vectors and gated MLP biases.

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
