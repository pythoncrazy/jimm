from typing import Any

import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib
from jax.typing import DTypeLike
from jaxtyping import Array, Float

from jimm.common.vit import VisionTransformerBase
from jimm.models.vit.sharding import ViTSharding


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
        rngs: rnglib.Rngs | None = None,
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        sharding: ViTSharding | None = None,
    ) -> None:
        """Initialize a Vision Transformer.

        Args:
            num_classes (int, optional): Number of output classes. Defaults to 1000.
            in_channels (int, optional): Number of input channels. Defaults to 3.
            img_size (int, optional): Size of the input image (assumed square). Defaults to 224.
            patch_size (int, optional): Size of each patch (assumed square). Defaults to 16.
            num_layers (int, optional): Number of transformer layers. Defaults to 12.
            num_heads (int, optional): Number of attention heads. Defaults to 12.
            mlp_dim (int, optional): Size of the MLP dimension. Defaults to 3072.
            hidden_size (int, optional): Size of the hidden dimension. Defaults to 768.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
            use_quick_gelu (bool, optional): Whether to use quickgelu instead of gelu. Defaults to False.
            use_gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
            do_classification (bool, optional): Whether to include the final classification head. Defaults to True.
            rngs (rnglib.Rngs | None, optional): Random number generator keys. If None, initializes to nnx.Rngs(0).
            dtype (DTypeLike, optional): Data type for computations. Defaults to jnp.float32.
            param_dtype (DTypeLike, optional): Data type for parameters. Defaults to jnp.float32.
            sharding (ViTSharding | None, optional): Sharding specification for parameters. Defaults to None.
        """
        if rngs is None:
            rngs = nnx.Rngs(0)
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
            sharding=sharding,
        )

        if self.do_classification:
            self.classifier = nnx.Linear(
                hidden_size,
                num_classes,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
                kernel_init=nnx.with_partitioning(
                    nnx.initializers.xavier_uniform(),
                    sharding.proj_kernel if sharding else (None, None),
                ),
                bias_init=nnx.with_partitioning(
                    nnx.initializers.zeros_init(),
                    sharding.proj_bias if sharding else (None,),
                ),
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
        rngs: rnglib.Rngs | None = None,
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        sharding: ViTSharding | None = None,
        use_gradient_checkpointing: bool = False,
    ) -> "VisionTransformer":
        """Load a pretrained Vision Transformer from a local path or HuggingFace Hub.

        Args:
            model_name_or_path (str): Path to local weights or HuggingFace model ID.
            use_pytorch (bool): Whether to load from PyTorch weights. Defaults to False.
            rngs (rnglib.Rngs | None): Random number generator keys. If None, initializes to nnx.Rngs(0).
            dtype (DTypeLike): Data type for computations. Defaults to jnp.float32.
            param_dtype (DTypeLike): Data type for parameters. Defaults to jnp.float32.
            sharding (ViTSharding | None): Sharding specification for parameters. Defaults to None.
            use_gradient_checkpointing (bool): Whether to use gradient checkpointing. Defaults to False.

        Returns:
            VisionTransformer: Initialized Vision Transformer with pretrained weights
        """
        from .params import load_from_pretrained

        return load_from_pretrained(cls, model_name_or_path, use_pytorch, rngs, dtype, param_dtype, sharding, use_gradient_checkpointing)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        rngs: rnglib.Rngs | None = None,
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        sharding: ViTSharding | None = None,
        use_gradient_checkpointing: bool = False,
    ) -> "VisionTransformer":
        """Create model from HuggingFace-compatible config dict.

        Args:
            config: Configuration dictionary in HuggingFace ViT format.
            rngs: Random number generator state. If None, initializes to nnx.Rngs(0).
            dtype: Data type for computations.
            param_dtype: Data type for parameters.
            sharding: Sharding specification for parameters.
            use_gradient_checkpointing: Enable gradient checkpointing.

        Returns:
            VisionTransformer with random weights.
        """
        if rngs is None:
            rngs = nnx.Rngs(0)
        num_classes = len(config["id2label"]) if "id2label" in config else config.get("num_labels", 1000)
        use_quick_gelu = config.get("hidden_act") == "quick_gelu"

        return cls(
            num_classes=num_classes,
            img_size=config["image_size"],
            patch_size=config["patch_size"],
            num_layers=config["num_hidden_layers"],
            num_heads=config["num_attention_heads"],
            mlp_dim=config["intermediate_size"],
            hidden_size=config["hidden_size"],
            use_quick_gelu=use_quick_gelu,
            use_gradient_checkpointing=use_gradient_checkpointing,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            sharding=sharding,
        )

    def save_pretrained(self, save_directory: str):
        """Save the model weights and config in HuggingFace format.

        Args:
            save_directory (str): Directory path where the model will be saved.
        """
        from .params import save_pretrained

        save_pretrained(self, save_directory)
