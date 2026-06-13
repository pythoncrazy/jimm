from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib
from jax.typing import DTypeLike
from jaxtyping import Array, Float

from jimm.common.sharding import ShardingSpec
from jimm.common.vit import VisionTransformerBase
from jimm.models.dinov2.sharding import Dinov2Sharding


class DINOv2Model(nnx.Module):
    """DINOv2 vision transformer feature extractor.

    Implements the DINOv2 architecture from "DINOv2: Learning Robust Visual Features
    without Supervision" (Oquab et al., 2023). Returns CLS token embeddings.
    Key difference from standard ViT: per-channel LayerScale applied to attention
    and MLP residuals in each block.
    """

    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        in_channels: int = 3,
        hidden_size: int = 384,
        num_layers: int = 12,
        num_heads: int = 6,
        mlp_dim: int = 1536,
        layerscale_value: float = 1.0,
        use_gradient_checkpointing: bool = False,
        attention_fn: Callable[..., Any] | None = None,
        rngs: rnglib.Rngs | None = None,
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        sharding: ShardingSpec = Dinov2Sharding,
    ) -> None:
        """Initialize DINOv2Model.

        Args:
            img_size (int, optional): Input image size (square). Defaults to 518.
            patch_size (int, optional): Patch size. Defaults to 14.
            in_channels (int, optional): Number of input channels. Defaults to 3.
            hidden_size (int, optional): Hidden dimension. Defaults to 384 (small).
            num_layers (int, optional): Number of transformer layers. Defaults to 12.
            num_heads (int, optional): Number of attention heads. Defaults to 6.
            mlp_dim (int, optional): MLP intermediate dimension. Defaults to 1536.
            layerscale_value (float, optional): Initial LayerScale value. Defaults to 1.0.
            use_gradient_checkpointing (bool, optional): Enable gradient checkpointing. Defaults to False.
            attention_fn (Callable[..., Any] | None, optional): Custom attention function. Defaults to None.
            rngs (rnglib.Rngs | None, optional): RNG state. Defaults to nnx.Rngs(0).
            dtype (DTypeLike, optional): Computation dtype. Defaults to jnp.float32.
            param_dtype (DTypeLike, optional): Parameter dtype. Defaults to jnp.float32.
            sharding (ShardingSpec, optional): Sharding specification. Defaults to ViTSharding.
        """
        if rngs is None:
            rngs = nnx.Rngs(0)
        self._original_config = None
        self.encoder = VisionTransformerBase(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            pooling_type="CLS",
            dropout_rate=0.0,
            use_quick_gelu=False,
            use_pre_norm=False,
            use_patch_bias=True,
            layernorm_epsilon=1e-6,
            use_layer_scale=True,
            layer_scale_init=layerscale_value,
            use_gradient_checkpointing=use_gradient_checkpointing,
            attention_fn=attention_fn,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            sharding=sharding,
        )

    def __call__(self, x: Float[Array, "batch height width channels"]) -> Float[Array, "batch hidden_size"]:
        """Run DINOv2 forward pass and return CLS token embedding.

        Args:
            x (Float[Array, "batch height width channels"]): Input images in BHWC format.

        Returns:
            Float[Array, "batch hidden_size"]: CLS token embeddings after final LayerNorm.
        """
        return self.encoder(x)

    def save_pretrained(self, save_directory: str) -> None:
        """Save model weights and config in HuggingFace format.

        Args:
            save_directory (str): Directory path where the model will be saved.
        """
        from .params import save_pretrained as _save_pretrained

        _save_pretrained(self, save_directory)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        use_pytorch: bool = False,
        rngs: rnglib.Rngs | None = None,
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        sharding: ShardingSpec = Dinov2Sharding,
        use_gradient_checkpointing: bool = False,
        attention_fn: Callable[..., Any] | None = None,
    ) -> "DINOv2Model":
        """Load a pretrained DINOv2 model from a local path or HuggingFace Hub.

        Args:
            model_name_or_path (str): Local directory or HuggingFace model ID (e.g. "facebook/dinov2-small").
            use_pytorch (bool, optional): Load from PyTorch weights. Defaults to False.
            rngs (rnglib.Rngs | None, optional): RNG state. Defaults to nnx.Rngs(0).
            dtype (DTypeLike, optional): Computation dtype. Defaults to jnp.float32.
            param_dtype (DTypeLike, optional): Parameter dtype. Defaults to jnp.float32.
            sharding (ShardingSpec, optional): Sharding specification. Defaults to ViTSharding.
            use_gradient_checkpointing (bool, optional): Enable gradient checkpointing. Defaults to False.
            attention_fn (Callable[..., Any] | None, optional): Custom attention function. Defaults to None.

        Returns:
            DINOv2Model: Model with pretrained weights loaded.
        """
        from .params import load_from_pretrained

        return load_from_pretrained(
            cls,
            model_name_or_path,
            use_pytorch,
            rngs,
            dtype,
            param_dtype,
            sharding,
            use_gradient_checkpointing,
            attention_fn,
        )

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        rngs: rnglib.Rngs | None = None,
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        sharding: ShardingSpec = Dinov2Sharding,
        use_gradient_checkpointing: bool = False,
        attention_fn: Callable[..., Any] | None = None,
    ) -> "DINOv2Model":
        """Create DINOv2Model from a HuggingFace-compatible config dict.

        Args:
            config (dict[str, Any]): Config dict in HuggingFace DINOv2 format.
            rngs (rnglib.Rngs | None, optional): RNG state. Defaults to nnx.Rngs(0).
            dtype (DTypeLike, optional): Computation dtype. Defaults to jnp.float32.
            param_dtype (DTypeLike, optional): Parameter dtype. Defaults to jnp.float32.
            sharding (ShardingSpec, optional): Sharding specification. Defaults to ViTSharding.
            use_gradient_checkpointing (bool, optional): Enable gradient checkpointing. Defaults to False.
            attention_fn (Callable[..., Any] | None, optional): Custom attention function. Defaults to None.

        Returns:
            DINOv2Model: Model with random weights matching the given config.
        """
        if rngs is None:
            rngs = nnx.Rngs(0)
        hidden_size = config["hidden_size"]
        mlp_dim = int(hidden_size * config.get("mlp_ratio", 4))
        return cls(
            img_size=config["image_size"],
            patch_size=config["patch_size"],
            in_channels=config.get("num_channels", 3),
            hidden_size=hidden_size,
            num_layers=config["num_hidden_layers"],
            num_heads=config["num_attention_heads"],
            mlp_dim=mlp_dim,
            layerscale_value=config.get("layerscale_value", 1.0),
            use_gradient_checkpointing=use_gradient_checkpointing,
            attention_fn=attention_fn,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            sharding=sharding,
        )
