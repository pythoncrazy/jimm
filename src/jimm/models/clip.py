import jax.numpy as jnp
from flax import nnx
from jaxtyping import DTypeLike
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jimm.common.transformer import Transformer
from jimm.common.utils import sharded_init
from jaxtyping import Array, Float


class VisionTransformer(nnx.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        rngs: nnx.Rngs = nnx.Rngs(0),
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        mesh: Mesh | None = None,
    ):
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nnx.Conv(
            in_features=3,
            out_features=width,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            kernel_init=sharded_init(nnx.initializers.xavier_uniform(), P(None, None, None, "model"), mesh),
            bias_init=sharded_init(nnx.initializers.zeros_init(), P("model"), mesh),
        )
        _cls_token_initializer = sharded_init(nnx.initializers.zeros_init(), P(None, None, "model"), mesh)
        cls_token_value: Float[Array, "one one width"] = _cls_token_initializer(rngs.params(), (1, 1, width), dtype=dtype)
        self.cls_token = nnx.Param(cls_token_value)
        self.positional_embedding


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
