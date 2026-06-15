import warnings
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib
from jax.typing import DTypeLike
from jaxtyping import Array, Float

from jimm.common.sharding import ShardingSpec
from jimm.common.vit import VisionTransformerBase
from jimm.models.dinov3.sharding import DINOv3Sharding


class DINOv3Model(nnx.Module):
    """DINOv3 vision transformer feature extractor.

    Implements the DINOv3 architecture with 2D rotary position embeddings (RoPE),
    register tokens, and optional gated MLP. Returns CLS token embeddings.
    Supports variable image sizes (any multiple of patch_size) via dynamic RoPE.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_size: int = 384,
        num_layers: int = 12,
        num_heads: int = 6,
        mlp_dim: int = 1536,
        num_register_tokens: int = 4,
        rope_theta: float = 100.0,
        layer_scale_init: float = 1.0,
        layernorm_epsilon: float = 1e-5,
        hidden_act: str = "gelu",
        use_gated_mlp: bool = False,
        use_patch_bias: bool = True,
        use_gradient_checkpointing: bool = False,
        attention_fn: Callable[..., Any] | None = None,
        rngs: rnglib.Rngs | None = None,
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        sharding: ShardingSpec = DINOv3Sharding(),
    ) -> None:
        """Initialize DINOv3Model.

        Args:
            img_size (int, optional): Input image size hint (ignored at runtime — variable size supported). Defaults to 224.
            patch_size (int, optional): Patch size. Defaults to 16.
            in_channels (int, optional): Number of input channels. Defaults to 3.
            hidden_size (int, optional): Hidden dimension size. Defaults to 384 (small).
            num_layers (int, optional): Number of transformer layers. Defaults to 12.
            num_heads (int, optional): Number of attention heads. Defaults to 6.
            mlp_dim (int, optional): MLP intermediate dimension. Defaults to 1536.
            num_register_tokens (int, optional): Number of learnable register tokens. Defaults to 4.
            rope_theta (float, optional): RoPE base frequency. Defaults to 100.0.
            layer_scale_init (float, optional): Initial value for per-channel LayerScale parameters. Defaults to 1.0.
            layernorm_epsilon (float, optional): LayerNorm epsilon. Defaults to 1e-5.
            hidden_act (str, optional): MLP activation — "gelu" or "silu". Defaults to "gelu".
            use_gated_mlp (bool, optional): Whether to use gated (SwiGLU-style) MLP. Defaults to False.
            use_gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
            attention_fn (Callable[..., Any] | None, optional): Custom attention function. Note: DINOv3 uses a
                manual RoPE attention path that bypasses nnx.MultiHeadAttention.__call__, so attention_fn is
                not applied. Defaults to None.
            rngs (rnglib.Rngs | None, optional): The random number generator state. If None, initializes to nnx.Rngs(0).
            dtype (DTypeLike, optional): The data type for computations. Defaults to jnp.float32.
            param_dtype (DTypeLike, optional): The data type for parameters. Defaults to jnp.float32.
            sharding (ShardingSpec, optional): Sharding specification for parameters. Defaults to DINOv3Sharding.
        """
        if rngs is None:
            rngs = nnx.Rngs(0)
        if attention_fn is not None:
            warnings.warn("attention_fn is ignored in the RoPE attention path", UserWarning, stacklevel=2)
        self._original_config = None
        self._layer_scale_init = layer_scale_init
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
            use_patch_bias=use_patch_bias,
            layernorm_epsilon=layernorm_epsilon,
            use_layer_scale=True,
            layer_scale_init=layer_scale_init,
            use_rope=True,
            rope_theta=rope_theta,
            num_register_tokens=num_register_tokens,
            use_gated_mlp=use_gated_mlp,
            hidden_act=hidden_act,
            key_bias=False,
            use_gradient_checkpointing=use_gradient_checkpointing,
            attention_fn=attention_fn,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            sharding=sharding,
        )

    def __call__(self, x: Float[Array, "batch height width channels"]) -> Float[Array, "batch hidden_size"]:
        """Run DINOv3 forward pass and return CLS token embedding.

        Args:
            x (Float[Array, "batch height width channels"]): Batch of input images in BHWC format.
                Height and width must each be divisible by patch_size.

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
        sharding: ShardingSpec = DINOv3Sharding(),
        use_gradient_checkpointing: bool = False,
        attention_fn: Callable[..., Any] | None = None,
    ) -> "DINOv3Model":
        """Load a pretrained DINOv3 model from a local path or HuggingFace Hub.

        Args:
            model_name_or_path (str): Local directory or HuggingFace model ID (e.g. "facebook/dinov3-vits16-pretrain-lvd1689m").
            use_pytorch (bool): Whether to load from PyTorch weights. Defaults to False.
            rngs (rnglib.Rngs | None): The random number generator state. If None, initializes to nnx.Rngs(0).
            dtype (DTypeLike): The data type for computations. Defaults to jnp.float32.
            param_dtype (DTypeLike): The data type for parameters. Defaults to jnp.float32.
            sharding (ShardingSpec): Sharding specification for parameters. Defaults to DINOv3Sharding.
            use_gradient_checkpointing (bool): Whether to use gradient checkpointing. Defaults to False.
            attention_fn (Callable[..., Any] | None): Custom attention function. Defaults to None.

        Returns:
            DINOv3Model: Model with pretrained weights loaded.
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
        hidden_size = config["hidden_size"]
        mlp_dim = config.get("intermediate_size", int(hidden_size * config.get("mlp_ratio", 4)))
        return {
            "img_size": config.get("image_size", 224),
            "patch_size": config["patch_size"],
            "in_channels": config.get("num_channels", 3),
            "hidden_size": hidden_size,
            "num_layers": config["num_hidden_layers"],
            "num_heads": config["num_attention_heads"],
            "mlp_dim": mlp_dim,
            "num_register_tokens": config.get("num_register_tokens", 4),
            "rope_theta": config.get("rope_theta", 100.0),
            "layer_scale_init": config.get("layerscale_value", 1.0),
            "layernorm_epsilon": config.get("layer_norm_eps", 1e-5),
            "hidden_act": config.get("hidden_act", "gelu"),
            "use_gated_mlp": config.get("use_gated_mlp", False),
            "use_patch_bias": config.get("use_patch_bias", True),
        }

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        rngs: rnglib.Rngs | None = None,
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        sharding: ShardingSpec = DINOv3Sharding(),
        use_gradient_checkpointing: bool = False,
        attention_fn: Callable[..., Any] | None = None,
    ) -> "DINOv3Model":
        """Create DINOv3Model from a HuggingFace-compatible config dict.

        Args:
            config (dict[str, Any]): Configuration dictionary in HuggingFace DINOv3 format.
            rngs (rnglib.Rngs | None): The random number generator state. If None, initializes to nnx.Rngs(0).
            dtype (DTypeLike): The data type for computations. Defaults to jnp.float32.
            param_dtype (DTypeLike): The data type for parameters. Defaults to jnp.float32.
            sharding (ShardingSpec): Sharding specification for parameters. Defaults to DINOv3Sharding.
            use_gradient_checkpointing (bool): Whether to use gradient checkpointing. Defaults to False.
            attention_fn (Callable[..., Any] | None): Custom attention function. Defaults to None.

        Returns:
            DINOv3Model: Model with randomly initialized weights matching the given config.
        """
        if rngs is None:
            rngs = nnx.Rngs(0)
        return cls(**cls._parse_config(config), use_gradient_checkpointing=use_gradient_checkpointing, attention_fn=attention_fn, rngs=rngs, dtype=dtype, param_dtype=param_dtype, sharding=sharding)
