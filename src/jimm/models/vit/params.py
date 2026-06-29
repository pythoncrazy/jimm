import json
import os
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Any, cast

import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib
from jax.typing import DTypeLike
from jaxtyping import Array
from safetensors.flax import save_file as save_safetensors

from jimm.common.loading_utils import (
    apply_mapping,
    expand_scanned_layers,
    load_params_and_config,
)
from jimm.common.sharding import ShardingSpec
from jimm.common.transformer import quickgelu
from jimm.common.utils import convert_key_to_hf_format, filter_tensors
from jimm.models.vit.sharding import ViTSharding

if TYPE_CHECKING:
    from jimm.models import VisionTransformer


def _get_key_and_transform_mapping() -> dict[str, tuple[str, Any]]:
    """Return regex-based key mapping from HuggingFace to Flax format for ViT.

    Returns:
        dict[str, tuple[str, Any]]: Dict of {regex_pattern: (flax_key_template, Transform)}.
    """

    class Transform(Enum):
        BIAS = (None, None, False)
        LINEAR = ((1, 0), None, False)
        CONV2D = ((2, 3, 1, 0), None, False)
        DEFAULT = (None, None, False)

    return {
        r"vit\.embeddings\.cls_token$": ("encoder.cls_token", Transform.DEFAULT),
        r"vit\.embeddings\.position_embeddings$": ("encoder.position_embeddings", Transform.DEFAULT),
        r"vit\.embeddings\.patch_embeddings\.projection\.weight$": ("encoder.patch_embeddings.kernel", Transform.CONV2D),
        r"vit\.embeddings\.patch_embeddings\.projection\.bias$": ("encoder.patch_embeddings.bias", Transform.BIAS),
        r"vit\.layernorm\.weight$": ("encoder.ln_post.scale", Transform.DEFAULT),
        r"vit\.layernorm\.bias$": ("encoder.ln_post.bias", Transform.BIAS),
        r"classifier\.weight$": ("classifier.kernel", Transform.LINEAR),
        r"classifier\.bias$": ("classifier.bias", Transform.BIAS),
        r"vit\.encoder\.layer\.([0-9]+)\.attention\.attention\.query\.weight$": (r"encoder.layers_\1.attn.query.kernel", Transform.LINEAR),
        r"vit\.encoder\.layer\.([0-9]+)\.attention\.attention\.key\.weight$": (r"encoder.layers_\1.attn.key.kernel", Transform.LINEAR),
        r"vit\.encoder\.layer\.([0-9]+)\.attention\.attention\.value\.weight$": (r"encoder.layers_\1.attn.value.kernel", Transform.LINEAR),
        r"vit\.encoder\.layer\.([0-9]+)\.attention\.attention\.query\.bias$": (r"encoder.layers_\1.attn.query.bias", Transform.BIAS),
        r"vit\.encoder\.layer\.([0-9]+)\.attention\.attention\.key\.bias$": (r"encoder.layers_\1.attn.key.bias", Transform.BIAS),
        r"vit\.encoder\.layer\.([0-9]+)\.attention\.attention\.value\.bias$": (r"encoder.layers_\1.attn.value.bias", Transform.BIAS),
        r"vit\.encoder\.layer\.([0-9]+)\.attention\.output\.dense\.weight$": (r"encoder.layers_\1.attn.out.kernel", Transform.LINEAR),
        r"vit\.encoder\.layer\.([0-9]+)\.attention\.output\.dense\.bias$": (r"encoder.layers_\1.attn.out.bias", Transform.BIAS),
        r"vit\.encoder\.layer\.([0-9]+)\.intermediate\.dense\.weight$": (r"encoder.layers_\1.mlp.layers.0.kernel", Transform.LINEAR),
        r"vit\.encoder\.layer\.([0-9]+)\.intermediate\.dense\.bias$": (r"encoder.layers_\1.mlp.layers.0.bias", Transform.BIAS),
        r"vit\.encoder\.layer\.([0-9]+)\.output\.dense\.weight$": (r"encoder.layers_\1.mlp.layers.3.kernel", Transform.LINEAR),
        r"vit\.encoder\.layer\.([0-9]+)\.output\.dense\.bias$": (r"encoder.layers_\1.mlp.layers.3.bias", Transform.BIAS),
        r"vit\.encoder\.layer\.([0-9]+)\.layernorm_before\.weight$": (r"encoder.layers_\1.norm1.scale", Transform.DEFAULT),
        r"vit\.encoder\.layer\.([0-9]+)\.layernorm_before\.bias$": (r"encoder.layers_\1.norm1.bias", Transform.BIAS),
        r"vit\.encoder\.layer\.([0-9]+)\.layernorm_after\.weight$": (r"encoder.layers_\1.norm2.scale", Transform.DEFAULT),
        r"vit\.encoder\.layer\.([0-9]+)\.layernorm_after\.bias$": (r"encoder.layers_\1.norm2.bias", Transform.BIAS),
    }


def _create_config(model: "VisionTransformer") -> dict[str, Any]:
    """Create HuggingFace-compatible config dictionary.

    Args:
        model (VisionTransformer): The VisionTransformer model instance.

    Returns:
        dict[str, Any]: Configuration dictionary in HuggingFace format.
    """
    if model._original_config is not None:
        return model._original_config.copy()

    patch_size = model.encoder.patch_embeddings.kernel_size[0]
    hidden_size = model.encoder.patch_embeddings.out_features
    num_heads = model.encoder.layers.attn.num_heads
    mlp_up = cast(nnx.Linear, model.encoder.layers.mlp.layers[0])
    mlp_dim = mlp_up.out_features
    n_patches_plus_one = model.encoder.position_embeddings[...].shape[1]
    img_size = int((n_patches_plus_one - 1) ** 0.5) * patch_size

    return {
        "model_type": "vit",
        "architectures": ["ViTForImageClassification"],
        "hidden_size": hidden_size,
        "num_hidden_layers": model.encoder.num_layers,
        "num_attention_heads": num_heads,
        "intermediate_size": mlp_dim,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "image_size": img_size,
        "patch_size": patch_size,
        "num_channels": model.encoder.patch_embeddings.in_features,
        "qkv_bias": True,
    }


def _convert_vit_tensor_to_hf_format(hf_key: str, tensor: Array, num_heads: int, hidden_size: int, head_dim: int) -> Array:
    """Convert ViT tensor from Flax format to HuggingFace format.

    Args:
        hf_key (str): The HuggingFace key for the tensor.
        tensor (Array): The tensor to convert.
        num_heads (int): Number of attention heads.
        hidden_size (int): Hidden dimension size.
        head_dim (int): Dimension of each attention head.

    Returns:
        Array: The converted tensor in HuggingFace format.
    """
    if ".attention.attention.query.weight" in hf_key or ".attention.attention.key.weight" in hf_key or ".attention.attention.value.weight" in hf_key:
        if tensor.ndim == 3 and tensor.shape == (hidden_size, num_heads, head_dim):
            tensor = tensor.reshape((hidden_size, hidden_size))
            tensor = jnp.transpose(tensor, (1, 0))
            return tensor
    elif ".attention.attention.query.bias" in hf_key or ".attention.attention.key.bias" in hf_key or ".attention.attention.value.bias" in hf_key:
        if tensor.ndim == 2 and tensor.shape == (num_heads, head_dim):
            return tensor.reshape((hidden_size,))
    elif ".attention.output.dense.weight" in hf_key:
        if tensor.ndim == 3 and tensor.shape == (num_heads, head_dim, hidden_size):
            tensor = tensor.reshape((hidden_size, hidden_size))
            tensor = jnp.transpose(tensor, (1, 0))
            return tensor
    elif "vit.embeddings.patch_embeddings.projection.weight" in hf_key:
        if tensor.ndim == 4:
            return jnp.transpose(tensor, (3, 2, 0, 1))
    elif hf_key.endswith(".weight") and tensor.ndim == 2:
        return jnp.transpose(tensor, (1, 0))
    return tensor


def save_pretrained(model: "VisionTransformer", save_directory: str) -> None:
    """Save the model weights and config in HuggingFace format.

    Args:
        model (VisionTransformer): The VisionTransformer model instance to save.
        save_directory (str): Directory path where the model will be saved.
    """
    _SPECIAL_MAPPINGS = {
        "encoder.cls_token": "vit.embeddings.cls_token",
        "encoder.position_embeddings": "vit.embeddings.position_embeddings",
        "encoder.patch_embeddings.weight": "vit.embeddings.patch_embeddings.projection.weight",
        "encoder.patch_embeddings.bias": "vit.embeddings.patch_embeddings.projection.bias",
        "classifier.weight": "classifier.weight",
        "classifier.bias": "classifier.bias",
        "encoder.ln_post.weight": "vit.layernorm.weight",
        "encoder.ln_post.bias": "vit.layernorm.bias",
    }

    _SPECIAL_RENAMINGS: dict[str, str] = {
        ".attn.query.": ".attention.attention.query.",
        ".attn.key.": ".attention.attention.key.",
        ".attn.value.": ".attention.attention.value.",
        ".attn.out.": ".attention.output.dense.",
        ".mlp.layers.0.": ".intermediate.dense.",
        ".mlp.layers.3.": ".output.dense.",
        ".norm1.": ".layernorm_before.",
        ".norm2.": ".layernorm_after.",
    }
    for i in range(100):
        _SPECIAL_RENAMINGS[f"encoder.layers_{i}."] = f"vit.encoder.layer.{i}."

    os.makedirs(save_directory, exist_ok=True)

    config = _create_config(model)
    with open(os.path.join(save_directory, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    _, state = nnx.split(model)
    state_dict = nnx.to_pure_dict(state)
    state_dict["encoder"] = expand_scanned_layers(state_dict["encoder"])

    num_heads = model.encoder.layers.attn.num_heads
    hidden_size = model.encoder.patch_embeddings.out_features
    head_dim = hidden_size // num_heads

    tensor_state = filter_tensors(state_dict)
    hf_state = {}

    for jimm_key, tensor in tensor_state.items():
        hf_key = convert_key_to_hf_format(jimm_key, _SPECIAL_MAPPINGS, _SPECIAL_RENAMINGS)
        hf_tensor = _convert_vit_tensor_to_hf_format(hf_key, tensor, num_heads, hidden_size, head_dim)
        hf_state[hf_key] = hf_tensor

    save_safetensors(hf_state, os.path.join(save_directory, "model.safetensors"))


def load_from_pretrained(
    cls,
    model_name_or_path: str,
    use_pytorch: bool = False,
    rngs: rnglib.Rngs | None = None,
    dtype: DTypeLike = jnp.float32,
    param_dtype: DTypeLike = jnp.float32,
    sharding: ShardingSpec = ViTSharding,
    use_gradient_checkpointing: bool = False,
    attention_fn: Callable[..., Any] | None = None,
) -> "VisionTransformer":
    """Load a pretrained Vision Transformer from a local path or HuggingFace Hub.

    Args:
        cls: The VisionTransformer class.
        model_name_or_path (str): Path to local weights or HuggingFace model ID.
        use_pytorch (bool): Whether to load from PyTorch weights. Defaults to False.
        rngs (rnglib.Rngs | None): Random number generator keys. If None, initializes to nnx.Rngs(0).
        dtype (DTypeLike): Data type for computations. Defaults to jnp.float32.
        param_dtype (DTypeLike): Data type for parameters. Defaults to jnp.float32.
        sharding (ShardingSpec): Sharding specification for parameters. Defaults to ViTSharding.
        use_gradient_checkpointing (bool): Whether to use gradient checkpointing. Defaults to False.
        attention_fn (Callable[..., Any] | None): Custom attention function. Defaults to None.

    Returns:
        VisionTransformer: Initialized Vision Transformer with pretrained weights.
    """
    if rngs is None:
        rngs = nnx.Rngs(0)

    params_fstate, config_dict = load_params_and_config(model_name_or_path, use_pytorch)
    config: dict[str, Any] = config_dict

    if config:
        hidden_size = config["hidden_size"]
        num_classes = len(config["id2label"]) if "id2label" in config else config.get("num_labels", 1000)
        num_layers = config["num_hidden_layers"]
        num_heads = config["num_attention_heads"]
        mlp_dim = config["intermediate_size"]
        patch_size = config["patch_size"]
        img_size = config["image_size"]
        act_fn = quickgelu if config.get("hidden_act") == "quick_gelu" else None
    elif not use_pytorch and os.path.isfile(model_name_or_path):
        hidden_size = params_fstate["vit.embeddings.cls_token"].shape[-1]
        num_classes = params_fstate["classifier.bias"].shape[0]
        num_layers = max(int(k.split(".")[3]) + 1 for k in params_fstate if k.startswith("vit.encoder.layer.") and k.endswith(".bias"))
        mlp_dim = params_fstate["vit.encoder.layer.0.intermediate.dense.weight"].shape[0]
        num_heads = hidden_size // 64
        patch_size = params_fstate["vit.embeddings.patch_embeddings.projection.weight"].shape[2]
        n_patches = params_fstate["vit.embeddings.position_embeddings"].shape[1] - 1
        img_size = int(n_patches**0.5) * patch_size
        act_fn = None
    else:
        raise ValueError(f"Could not load or infer configuration for {model_name_or_path}")

    model = cls(
        num_classes=num_classes,
        img_size=img_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        hidden_size=hidden_size,
        act_fn=act_fn,
        sharding=sharding,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
        use_gradient_checkpointing=use_gradient_checkpointing,
        attention_fn=attention_fn,
    )

    apply_mapping(model, params_fstate, _get_key_and_transform_mapping(), param_dtype)
    model.eval()
    model._original_config = config_dict
    return model
