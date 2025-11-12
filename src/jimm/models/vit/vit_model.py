import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax.typing import DTypeLike
from jaxtyping import Array, Float

from jimm.common.utils import sharded_init
from jimm.common.vit import VisionTransformerBase


class VisionTransformer(nnx.Module):
    """Vision Transformer (ViT) model for image classification.

    This implements the Vision Transformer as described in the paper
    "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
    """

    def __init__(
        self,
        num_classes: int = 1000,
        in_channels: int = 3,
        img_size: int = 224,
        patch_size: int = 16,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        hidden_size: int = 768,
        dropout_rate: float = 0.1,
        use_quick_gelu: bool = False,
        use_gradient_checkpointing: bool = False,
        do_classification: bool = True,
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        rngs: rnglib.Rngs = nnx.Rngs(0),
        mesh: Mesh | None = None,
    ) -> None:
        """Initialize a Vision Transformer.

        Args:
            num_classes (int): Number of output classes. Defaults to 1000.
            in_channels (int): Number of input channels. Defaults to 3.
            img_size (int): Size of the input image (assumed square). Defaults to 224.
            patch_size (int): Size of each patch (assumed square). Defaults to 16.
            num_layers (int): Number of transformer layers. Defaults to 12.
            num_heads (int): Number of attention heads. Defaults to 12.
            mlp_dim (int): Size of the MLP dimension. Defaults to 3072.
            hidden_size (int): Size of the hidden dimension. Defaults to 768.
            dropout_rate (float): Dropout rate. Defaults to 0.1.
            use_quick_gelu (bool): Whether to use quickgelu instead of gelu. Defaults to False.
            use_gradient_checkpointing (bool): Whether to use gradient checkpointing. Defaults to False.
            do_classification (bool): Whether to include the final classification head. Defaults to True.
            dtype (DTypeLike): Data type for computations. Defaults to jnp.float32.
            param_dtype (DTypeLike): Data type for parameters. Defaults to jnp.float32.
            rngs (rnglib.Rngs): Random number generator keys. Defaults to nnx.Rngs(0).
            mesh (Mesh | None): Optional JAX device mesh for parameter sharding. Defaults to None.
        """
        self.do_classification = do_classification
        self._original_config = None
        self.encoder = VisionTransformerBase(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout_rate=dropout_rate,
            use_quick_gelu=use_quick_gelu,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_pre_norm=False,
            use_patch_bias=True,
            layernorm_epsilon=1e-12,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            mesh=mesh,
        )

        if self.do_classification:
            self.classifier = nnx.Linear(
                hidden_size,
                num_classes,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
                kernel_init=sharded_init(nnx.initializers.xavier_uniform(), P(None, "fsdp"), mesh),
                bias_init=sharded_init(nnx.initializers.zeros_init(), P("fsdp"), mesh),
            )

    def __call__(self, x: Float[Array, "batch height width channels"]) -> Float[Array, "batch num_classes"]:
        """Forward pass of the Vision Transformer.

        Args:
            x (Float[Array, "batch height width channels"]): Input tensor with shape [batch, height, width, channels]

        Returns:
            Float[Array, "batch num_classes"]: Output logits with shape [batch, num_classes]
        """
        x = self.encoder(x)
        if self.do_classification:
            return self.classifier(x)
        return x

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        use_pytorch: bool = False,
        mesh: Mesh | None = None,
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        use_gradient_checkpointing: bool = False,
        rngs: rnglib.Rngs = nnx.Rngs(0),
    ) -> "VisionTransformer":
        """Load a pretrained Vision Transformer from a local path or HuggingFace Hub.

        Args:
            model_name_or_path (str): Path to local weights or HuggingFace model ID.
            use_pytorch (bool): Whether to load from PyTorch weights. Defaults to False.
            mesh (Mesh | None): Optional device mesh for parameter sharding. Defaults to None.
            dtype (DTypeLike): Data type for computations. Defaults to jnp.float32.
            param_dtype (DTypeLike): Data type for parameters. Defaults to jnp.float32.
            use_gradient_checkpointing (bool): Whether to use gradient checkpointing. Defaults to False.
            rngs (rnglib.Rngs): Random number generator keys. Defaults to nnx.Rngs(0).

        Returns:
            VisionTransformer: Initialized Vision Transformer with pretrained weights
        """
        from .params import load_from_pretrained

        return load_from_pretrained(cls, model_name_or_path, use_pytorch, mesh, dtype, param_dtype, use_gradient_checkpointing, rngs)

    def save_pretrained(self, save_directory: str):
        """Save the model weights and config in HuggingFace format.

        Args:
            save_directory (str): Directory path where the model will be saved.
        """
        from .params import save_pretrained

        save_pretrained(self, save_directory)
