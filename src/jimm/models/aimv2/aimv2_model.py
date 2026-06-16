from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib
from jax.typing import DTypeLike
from jaxtyping import Array, Float

from jimm.common.sharding import ShardingSpec
from jimm.common.vit import VisionTransformerBase
from jimm.models.aimv2.sharding import AIMv2Sharding


class AIMv2Model(nnx.Module):
    """AIMv2 vision encoder.

    Implements the AIMv2 vision transformer with RMSNorm, SwiGLU MLP, no
    attention/MLP biases, and absolute learnable positional embeddings.
    Returns all patch token embeddings (no CLS token).
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 14,
        in_channels: int = 3,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 6,
        mlp_dim: int = 2048,
        rms_norm_eps: float = 1e-5,
        use_gradient_checkpointing: bool = False,
        attention_fn: Callable[..., Any] | None = None,
        rngs: rnglib.Rngs | None = None,
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        sharding: ShardingSpec = AIMv2Sharding(),
    ) -> None:
        """Initialize AIMv2Model.

        Args:
            img_size (int, optional): Input image size. Defaults to 224.
            patch_size (int, optional): Patch size. Defaults to 14.
            in_channels (int, optional): Number of input channels. Defaults to 3.
            hidden_size (int, optional): Hidden dimension. Defaults to 768 (base).
            num_layers (int, optional): Number of transformer layers. Defaults to 12.
            num_heads (int, optional): Number of attention heads. Defaults to 6.
            mlp_dim (int, optional): MLP intermediate dimension. Defaults to 2048.
            rms_norm_eps (float, optional): RMSNorm epsilon. Defaults to 1e-5.
            use_gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
            attention_fn (Callable[..., Any] | None, optional): Custom attention function. Defaults to None.
            rngs (rnglib.Rngs | None, optional): RNG state. If None, initializes to nnx.Rngs(0).
            dtype (DTypeLike, optional): Computation dtype. Defaults to jnp.float32.
            param_dtype (DTypeLike, optional): Parameter dtype. Defaults to jnp.float32.
            sharding (ShardingSpec, optional): Sharding specification. Defaults to AIMv2Sharding.
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
            layernorm_epsilon=rms_norm_eps,
            pooling_type="ALL",
            dropout_rate=0.0,
            use_quick_gelu=False,
            use_pre_norm=True,
            pre_norm_before_pos=True,
            use_patch_bias=True,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_layer_scale=False,
            use_rope=False,
            num_register_tokens=0,
            use_gated_mlp=True,
            hidden_act="silu",
            key_bias=False,
            attn_bias=False,
            mlp_bias=False,
            use_rms_norm=True,
            attention_fn=attention_fn,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            sharding=sharding,
        )

    def __call__(self, x: Float[Array, "batch height width channels"]) -> Float[Array, "batch n_patches hidden_size"]:
        """Run AIMv2 forward pass and return all patch token embeddings.

        Args:
            x (Float[Array, "batch height width channels"]): Batch of input images in BHWC format.
                Height and width must each be divisible by patch_size.

        Returns:
            Float[Array, "batch n_patches hidden_size"]: Patch token embeddings after final RMSNorm.
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
        sharding: ShardingSpec = AIMv2Sharding(),
        use_gradient_checkpointing: bool = False,
        attention_fn: Callable[..., Any] | None = None,
    ) -> "AIMv2Model":
        """Load a pretrained AIMv2 model from a local path or HuggingFace Hub.

        Args:
            model_name_or_path (str): Local directory or HuggingFace model ID
                (e.g. "apple/aimv2-large-patch14-224").
            use_pytorch (bool): Whether to load from PyTorch weights. Defaults to False.
            rngs (rnglib.Rngs | None): RNG state. If None, initializes to nnx.Rngs(0).
            dtype (DTypeLike): Computation dtype. Defaults to jnp.float32.
            param_dtype (DTypeLike): Parameter dtype. Defaults to jnp.float32.
            sharding (ShardingSpec): Sharding specification. Defaults to AIMv2Sharding.
            use_gradient_checkpointing (bool): Whether to use gradient checkpointing. Defaults to False.
            attention_fn (Callable[..., Any] | None): Custom attention function. Defaults to None.

        Returns:
            AIMv2Model: Model with pretrained weights loaded.
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
    def _parse_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        if config.get("model_type") == "aimv2":
            config = config["vision_config"]
        return {
            "img_size": config.get("image_size", 224),
            "patch_size": config.get("patch_size", 14),
            "in_channels": config.get("num_channels", 3),
            "hidden_size": config["hidden_size"],
            "num_layers": config["num_hidden_layers"],
            "num_heads": config["num_attention_heads"],
            "mlp_dim": config["intermediate_size"],
            "rms_norm_eps": config.get("rms_norm_eps", 1e-5),
        }

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        rngs: rnglib.Rngs | None = None,
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        sharding: ShardingSpec = AIMv2Sharding(),
        use_gradient_checkpointing: bool = False,
        attention_fn: Callable[..., Any] | None = None,
    ) -> "AIMv2Model":
        """Create AIMv2Model from a HuggingFace-compatible config dict.

        Args:
            config (dict[str, Any]): Configuration dictionary in HuggingFace AIMv2 format.
            rngs (rnglib.Rngs | None): RNG state. If None, initializes to nnx.Rngs(0).
            dtype (DTypeLike): Computation dtype. Defaults to jnp.float32.
            param_dtype (DTypeLike): Parameter dtype. Defaults to jnp.float32.
            sharding (ShardingSpec): Sharding specification. Defaults to AIMv2Sharding.
            use_gradient_checkpointing (bool): Whether to use gradient checkpointing. Defaults to False.
            attention_fn (Callable[..., Any] | None): Custom attention function. Defaults to None.

        Returns:
            AIMv2Model: Model with randomly initialized weights matching the given config.
        """
        if rngs is None:
            rngs = nnx.Rngs(0)
        return cls(
            **cls._parse_config(config),
            use_gradient_checkpointing=use_gradient_checkpointing,
            attention_fn=attention_fn,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            sharding=sharding,
        )
