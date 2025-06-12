import jax.numpy as jnp
from flax import nnx
from jaxtyping import DTypeLike


class VisionTransformer(nnx.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        self.input_resolution = input_resolution


class CLIP(nnx.Module):
    def __init__(
        self,
        # Vision
        image_resolution: int,
        vision_layers: int,
        vision_width: int,
        vision_patch_size: int,
        # Text
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        dtype: DTypeLike = jnp.float32,
    ):
        self.vision_layers = vision_layers
        self.vision_width = vision_width
        self.vision_patch_size = vision_patch_size
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.transformer_width = transformer_width
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers
        self.dtype = dtype

        vision_heads = vision_width // 64
        # self.vision_model = VisionTransformer(
        #     num_classes=0,
        #     in_channels=3,
        #     img_size=image_resolution,
        #     patch_size=vision_patch_size,
        #     num_layers=vision_layers,
        #     num_heads=vision_heads,
        #     mlp_dim=vision_width * 4,
        #     hidden_size=vision_width,
        #     dropout_rate=0.0,
        #     dtype=dtype,
        # )
