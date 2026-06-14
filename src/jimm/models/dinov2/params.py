import json
import os
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib
from jax.typing import DTypeLike
from jaxtyping import Array
from safetensors.flax import save_file as save_safetensors

from jimm.common.loading_utils import apply_mapping, expand_scanned_layers, load_params_and_config
from jimm.common.sharding import ShardingSpec
from jimm.common.utils import convert_key_to_hf_format, filter_tensors
from jimm.models.dinov2.sharding import DINOv2Sharding

if TYPE_CHECKING:
    from jimm.models.dinov2.dinov2_model import DINOv2Model


def _get_key_and_transform_mapping() -> dict[str, tuple[str, Any]]:
    """Return regex-based key mapping from HuggingFace to Flax format for DINOv2.

    Returns:
        dict[str, tuple[str, Any]]: Dict of {regex_pattern: (flax_key_template, Transform)}.
    """

    class Transform(Enum):
        BIAS = (None, None, False)
        LINEAR = ((1, 0), None, False)
        CONV2D = ((2, 3, 1, 0), None, False)
        DEFAULT = (None, None, False)

    return {
        r"embeddings\.cls_token$": ("encoder.cls_token", Transform.DEFAULT),
        r"embeddings\.position_embeddings$": ("encoder.position_embeddings", Transform.DEFAULT),
        r"embeddings\.patch_embeddings\.projection\.weight$": ("encoder.patch_embeddings.kernel", Transform.CONV2D),
        r"embeddings\.patch_embeddings\.projection\.bias$": ("encoder.patch_embeddings.bias", Transform.BIAS),
        r"layernorm\.weight$": ("encoder.ln_post.scale", Transform.DEFAULT),
        r"layernorm\.bias$": ("encoder.ln_post.bias", Transform.BIAS),
        r"encoder\.layer\.([0-9]+)\.attention\.attention\.query\.weight$": (r"encoder.layers_\1.attn.query.kernel", Transform.LINEAR),
        r"encoder\.layer\.([0-9]+)\.attention\.attention\.key\.weight$": (r"encoder.layers_\1.attn.key.kernel", Transform.LINEAR),
        r"encoder\.layer\.([0-9]+)\.attention\.attention\.value\.weight$": (r"encoder.layers_\1.attn.value.kernel", Transform.LINEAR),
        r"encoder\.layer\.([0-9]+)\.attention\.attention\.query\.bias$": (r"encoder.layers_\1.attn.query.bias", Transform.BIAS),
        r"encoder\.layer\.([0-9]+)\.attention\.attention\.key\.bias$": (r"encoder.layers_\1.attn.key.bias", Transform.BIAS),
        r"encoder\.layer\.([0-9]+)\.attention\.attention\.value\.bias$": (r"encoder.layers_\1.attn.value.bias", Transform.BIAS),
        r"encoder\.layer\.([0-9]+)\.attention\.output\.dense\.weight$": (r"encoder.layers_\1.attn.out.kernel", Transform.LINEAR),
        r"encoder\.layer\.([0-9]+)\.attention\.output\.dense\.bias$": (r"encoder.layers_\1.attn.out.bias", Transform.BIAS),
        r"encoder\.layer\.([0-9]+)\.layer_scale1\.lambda1$": (r"encoder.layers_\1.layer_scale1", Transform.DEFAULT),
        r"encoder\.layer\.([0-9]+)\.layer_scale2\.lambda1$": (r"encoder.layers_\1.layer_scale2", Transform.DEFAULT),
        r"encoder\.layer\.([0-9]+)\.mlp\.fc1\.weight$": (r"encoder.layers_\1.mlp.layers.0.kernel", Transform.LINEAR),
        r"encoder\.layer\.([0-9]+)\.mlp\.fc1\.bias$": (r"encoder.layers_\1.mlp.layers.0.bias", Transform.BIAS),
        r"encoder\.layer\.([0-9]+)\.mlp\.fc2\.weight$": (r"encoder.layers_\1.mlp.layers.3.kernel", Transform.LINEAR),
        r"encoder\.layer\.([0-9]+)\.mlp\.fc2\.bias$": (r"encoder.layers_\1.mlp.layers.3.bias", Transform.BIAS),
        r"encoder\.layer\.([0-9]+)\.norm1\.weight$": (r"encoder.layers_\1.norm1.scale", Transform.DEFAULT),
        r"encoder\.layer\.([0-9]+)\.norm1\.bias$": (r"encoder.layers_\1.norm1.bias", Transform.BIAS),
        r"encoder\.layer\.([0-9]+)\.norm2\.weight$": (r"encoder.layers_\1.norm2.scale", Transform.DEFAULT),
        r"encoder\.layer\.([0-9]+)\.norm2\.bias$": (r"encoder.layers_\1.norm2.bias", Transform.BIAS),
    }


def _create_dinov2_config(model: "DINOv2Model") -> dict[str, Any]:
    """Create HuggingFace-compatible config dictionary for DINOv2.

    Args:
        model (DINOv2Model): The DINOv2Model instance.

    Returns:
        dict[str, Any]: Configuration dictionary in HuggingFace DINOv2 format.
    """
    if model._original_config is not None:
        return model._original_config.copy()

    patch_size = model.encoder.patch_embeddings.kernel_size[0]
    hidden_size = model.encoder.patch_embeddings.out_features
    num_heads = model.encoder.num_heads
    mlp_dim = model.encoder.mlp_dim
    n_patches_plus_one = model.encoder.position_embeddings[...].shape[1]
    img_size = int((n_patches_plus_one - 1) ** 0.5) * patch_size

    return {
        "model_type": "dinov2",
        "architectures": ["Dinov2Model"],
        "hidden_size": hidden_size,
        "num_hidden_layers": model.encoder.num_layers,
        "num_attention_heads": num_heads,
        "mlp_ratio": mlp_dim / hidden_size,
        "hidden_act": "gelu",
        "layerscale_value": model._layerscale_value,
        "drop_path_rate": 0.0,
        "layer_norm_eps": 1e-6,
        "image_size": img_size,
        "patch_size": patch_size,
        "num_channels": model.encoder.patch_embeddings.in_features,
        "qkv_bias": True,
        "initializer_range": 0.02,
    }


def _convert_dinov2_tensor_to_hf_format(hf_key: str, tensor: Array) -> Array:
    """Convert DINOv2 tensor from Flax format to HuggingFace format.

    Args:
        hf_key (str): The HuggingFace key for the tensor.
        tensor (Array): The tensor to convert.

    Returns:
        Array: The converted tensor in HuggingFace format.
    """
    if ".attention.attention.query.weight" in hf_key or ".attention.attention.key.weight" in hf_key or ".attention.attention.value.weight" in hf_key:
        if tensor.ndim == 3:
            return jnp.transpose(tensor.reshape(tensor.shape[0], -1), (1, 0))
    elif ".attention.attention.query.bias" in hf_key or ".attention.attention.key.bias" in hf_key or ".attention.attention.value.bias" in hf_key:
        if tensor.ndim == 2:
            return tensor.flatten()
    elif ".attention.output.dense.weight" in hf_key:
        if tensor.ndim == 3:
            return jnp.transpose(tensor.reshape(-1, tensor.shape[2]), (1, 0))
    elif "embeddings.patch_embeddings.projection.weight" in hf_key:
        if tensor.ndim == 4:
            return jnp.transpose(tensor, (3, 2, 0, 1))
    elif hf_key.endswith(".weight") and tensor.ndim == 2:
        return jnp.transpose(tensor, (1, 0))
    return tensor


def save_pretrained(model: "DINOv2Model", save_directory: str) -> None:
    """Save the DINOv2 model weights and config in HuggingFace format.

    Args:
        model (DINOv2Model): The DINOv2Model instance to save.
        save_directory (str): Directory path where the model will be saved.
    """
    _SPECIAL_MAPPINGS = {
        "encoder.cls_token": "embeddings.cls_token",
        "encoder.position_embeddings": "embeddings.position_embeddings",
        "encoder.patch_embeddings.weight": "embeddings.patch_embeddings.projection.weight",
        "encoder.patch_embeddings.bias": "embeddings.patch_embeddings.projection.bias",
        "encoder.ln_post.weight": "layernorm.weight",
        "encoder.ln_post.bias": "layernorm.bias",
    }
    _SPECIAL_RENAMINGS: dict[str, str] = {
        ".mlp.layers.0.": ".mlp.fc1.",
        ".mlp.layers.3.": ".mlp.fc2.",
        ".attn.query.": ".attention.attention.query.",
        ".attn.key.": ".attention.attention.key.",
        ".attn.value.": ".attention.attention.value.",
        ".attn.out.": ".attention.output.dense.",
        ".layer_scale1": ".layer_scale1.lambda1",
        ".layer_scale2": ".layer_scale2.lambda1",
    }
    for i in range(model.encoder.num_layers):
        _SPECIAL_RENAMINGS[f"encoder.layers_{i}."] = f"encoder.layer.{i}."

    os.makedirs(save_directory, exist_ok=True)

    config = _create_dinov2_config(model)
    with open(os.path.join(save_directory, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    _, state = nnx.split(model)
    state_dict = nnx.to_pure_dict(state)
    state_dict["encoder"] = expand_scanned_layers(state_dict["encoder"])

    tensor_state = filter_tensors(state_dict)
    hf_state: dict[str, Array] = {}
    for jimm_key, tensor in tensor_state.items():
        hf_key = convert_key_to_hf_format(jimm_key, _SPECIAL_MAPPINGS, _SPECIAL_RENAMINGS)
        hf_state[hf_key] = _convert_dinov2_tensor_to_hf_format(hf_key, tensor)

    save_safetensors(hf_state, os.path.join(save_directory, "model.safetensors"))


def load_from_pretrained(
    cls,
    model_name_or_path: str,
    use_pytorch: bool = False,
    rngs: rnglib.Rngs | None = None,
    dtype: DTypeLike = jnp.float32,
    param_dtype: DTypeLike = jnp.float32,
    sharding: ShardingSpec = DINOv2Sharding,
    use_gradient_checkpointing: bool = False,
    attention_fn: Callable[..., Any] | None = None,
) -> "DINOv2Model":
    """Load a pretrained DINOv2 model from a local path or HuggingFace Hub.

    Args:
        cls: The DINOv2Model class.
        model_name_or_path (str): Local directory or HuggingFace model ID.
        use_pytorch (bool, optional): Load from PyTorch weights. Defaults to False.
        rngs (rnglib.Rngs | None, optional): RNG state. Defaults to nnx.Rngs(0).
        dtype (DTypeLike, optional): Computation dtype. Defaults to jnp.float32.
        param_dtype (DTypeLike, optional): Parameter dtype. Defaults to jnp.float32.
        sharding (ShardingSpec, optional): Sharding specification. Defaults to DINOv2Sharding.
        use_gradient_checkpointing (bool, optional): Enable gradient checkpointing. Defaults to False.
        attention_fn (Callable[..., Any] | None, optional): Custom attention function. Defaults to None.

    Returns:
        DINOv2Model: Model with pretrained weights loaded.
    """
    if rngs is None:
        rngs = nnx.Rngs(0)

    params_fstate, config_dict = load_params_and_config(model_name_or_path, use_pytorch)
    config: dict[str, Any] = config_dict

    hidden_size = config["hidden_size"]
    mlp_dim = int(hidden_size * config.get("mlp_ratio", 4))
    layerscale_value = config.get("layerscale_value", 1.0)

    model = cls(
        img_size=config["image_size"],
        patch_size=config["patch_size"],
        in_channels=config.get("num_channels", 3),
        hidden_size=hidden_size,
        num_layers=config["num_hidden_layers"],
        num_heads=config["num_attention_heads"],
        mlp_dim=mlp_dim,
        layer_scale_init=layerscale_value,
        use_gradient_checkpointing=use_gradient_checkpointing,
        attention_fn=attention_fn,
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
        sharding=sharding,
    )

    apply_mapping(model, params_fstate, _get_key_and_transform_mapping(), param_dtype)
    model.eval()
    model._original_config = config_dict
    return model
