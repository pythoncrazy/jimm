import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from jaxtyping import Array, DTypeLike, Float

from jimm.common.vit import VisionTransformerBase


class SigLIP(nnx.Module):
    def __init__(
        self,
        image_resolution: int,
        vision_layers: int,
        vision_width: int,
        vision_patch_size: int,
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        rngs: nnx.Rngs = nnx.Rngs(0),
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        mesh: Mesh | None = None,
    ):
        """
        Initialize the CLIP model.

        Args:
            image_resolution (int): The resolution of the input images.
            vision_layers (int): The number of layers in the vision transformer.
            vision_width (int): The width of the vision transformer.
            vision_patch_size (int): The patch size of the vision transformer.
            context_length (int): The length of the context.
            vocab_size (int): The size of the vocabulary.
            transformer_width (int): The width of the transformer.
            transformer_heads (int): The number of attention heads in the transformer.
            transformer_layers (int): The number of layers in the transformer.
            rngs (nnx.Rngs): The random number generator state. Defaults to nnx.Rngs(0).
            dtype (DTypeLike): The data type for computations. Defaults to jnp.float32.
            param_dtype (DTypeLike): The data type for parameters. Defaults to jnp.float32.
            mesh (Mesh | None): The device mesh for parameter sharding. Defaults to None.
        """
        self.vision_layers = vision_layers
        self.vision_width = vision_width
        self.vision_patch_size = vision_patch_size
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.transformer_width = transformer_width
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers
        self.dtype = dtype

        self.vision_heads = vision_width // 64
        self.vision_model = VisionTransformerBase(
            img_size=image_resolution,
            patch_size=vision_patch_size,
            in_channels=3,
            hidden_size=vision_width,
            num_layers=vision_layers,
            num_heads=self.vision_heads,
            mlp_dim=vision_width * 4,
            use_pre_norm=True,
            use_patch_bias=False,
            use_quick_gelu=True,
            pooling_type="MAP",
            layernorm_epsilon=1e-5,
            dtype=dtype,
            param_dtype=param_dtype,
            mesh=mesh,
            rngs=rngs,
        )

    def encode_image(self, image: Float[Array, "batch height width channels"]) -> Float[Array, "batch transformer_width"]:
        """
        Encode images into embeddings.

        Args:
            image (Float[Array, "batch height width channels"]): Batch of input images.

        Returns:
            Float[Array, "batch transformer_width"]: Image embeddings.
        """
        return self.vision_model(image)
