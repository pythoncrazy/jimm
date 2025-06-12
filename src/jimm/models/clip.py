import jax.numpy as jnp
from flax import nnx
from jaxtyping import DTypeLike
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jimm.common.transformer import Transformer
from jimm.common.utils import sharded_init
from jaxtyping import Array, Float


# Needed as the CLIP Vision Transformer has an extra layernorm compared to the other vision transformer
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
        n_patches: int = (input_resolution // patch_size) ** 2
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
        _position_embeddings_initializer = sharded_init(nnx.initializers.truncated_normal(stddev=0.02), P(None, None, "model"), mesh)
        pos_emb_value: Float[Array, "one n_patches_plus_1 hidden_size_dim"] = _position_embeddings_initializer(rngs.params(), (1, n_patches + 1, width), dtype=dtype)
        self.position_embeddings = nnx.Param(pos_emb_value)

        self.ln_pre = nnx.LayerNorm(
            width,
            epsilon=1e-5,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            scale_init=sharded_init(nnx.initializers.ones_init(), P("model"), mesh),
            bias_init=sharded_init(nnx.initializers.zeros_init(), P("model"), mesh),
        )
        self.transformer = Transformer(
            width=width,
            mlp_dim=width * 4,
            layers=layers,
            heads=heads,
            dropout_rate=0.0,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            mesh=mesh,
        )
        self.ln_post = nnx.LayerNorm(
            width,
            epsilon=1e-5,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            scale_init=sharded_init(nnx.initializers.ones_init(), P("model"), mesh),
            bias_init=sharded_init(nnx.initializers.zeros_init(), P("model"), mesh),
        )

        self.proj = nnx.Linear(
            width,
            output_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            kernel_init=sharded_init(nnx.initializers.xavier_uniform(), P(None, "model"), mesh),
            bias_init=sharded_init(nnx.initializers.zeros_init(), P("model"), mesh),
        )
    def __call__(self, x: Float[Array, "batch height width channels"]) -> Float[Array, "batch output_dim"]:
        """
        Apply the CLIP vision transformer to input images.

        Args:
            x: Float[Array, "batch height width channels"]
                Batch of input images with shape (batch, height, width, channels).

        Returns:
            Float[Array, "batch output_dim"]
                Batch of output embeddings with shape (batch, output_dim).
        """
        # give proper typing to the below lines ai!
        x = self.conv1(x)
        x = x + self.position_embeddings
        x = jnp.concatenate([self.cls_token, x], axis=1)
        x = self.ln_pre(x)
        x = self.transformer(x)
        x = self.ln_post(x)
        x = self.proj(x)
        return x


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
