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

if TYPE_CHECKING:
    from jimm.models.aimv2.aimv2_model import AIMv2Model


class _Transform(Enum):
    LINEAR = ((1, 0), None, False)
    CONV2D = ((2, 3, 1, 0), None, False)
    DEFAULT = (None, None, False)


def _get_key_and_transform_mapping() -> dict[str, tuple[str, Any]]:
    """Return regex-based key mapping from HuggingFace to Flax format for AIMv2.

    Returns:
        dict[str, tuple[str, Any]]: Dict of {regex_pattern: (flax_key_template, _Transform)}.
    """
    return {
        r"embeddings\.patch_embed\.weight$": ("encoder.patch_embeddings.kernel", _Transform.CONV2D),
        r"embeddings\.patch_embed\.bias$": ("encoder.patch_embeddings.bias", _Transform.DEFAULT),
        r"embeddings\.position_embedding\.weight$": ("encoder.position_embeddings", _Transform.DEFAULT),
        r"embeddings\.rms_norm\.weight$": ("encoder.ln_pre.scale", _Transform.DEFAULT),
        r"encoder\.layers\.([0-9]+)\.rms_norm1\.weight$": (r"encoder.layers_\1.norm1.scale", _Transform.DEFAULT),
        r"encoder\.layers\.([0-9]+)\.rms_norm2\.weight$": (r"encoder.layers_\1.norm2.scale", _Transform.DEFAULT),
        r"encoder\.layers\.([0-9]+)\.attention\.q_proj\.weight$": (r"encoder.layers_\1.attn.query.kernel", _Transform.LINEAR),
        r"encoder\.layers\.([0-9]+)\.attention\.k_proj\.weight$": (r"encoder.layers_\1.attn.key.kernel", _Transform.LINEAR),
        r"encoder\.layers\.([0-9]+)\.attention\.v_proj\.weight$": (r"encoder.layers_\1.attn.value.kernel", _Transform.LINEAR),
        r"encoder\.layers\.([0-9]+)\.attention\.out_proj\.weight$": (r"encoder.layers_\1.attn.out.kernel", _Transform.LINEAR),
        r"encoder\.layers\.([0-9]+)\.ffn\.gate_proj\.weight$": (r"encoder.layers_\1.gate.kernel", _Transform.LINEAR),
        r"encoder\.layers\.([0-9]+)\.ffn\.up_proj\.weight$": (r"encoder.layers_\1.up.kernel", _Transform.LINEAR),
        r"encoder\.layers\.([0-9]+)\.ffn\.down_proj\.weight$": (r"encoder.layers_\1.down.kernel", _Transform.LINEAR),
        r"rms_norm\.weight$": ("encoder.ln_post.scale", _Transform.DEFAULT),
    }


def _create_aimv2_config(model: "AIMv2Model") -> dict[str, Any]:
    """Create HuggingFace-compatible config dictionary for AIMv2.

    Args:
        model (AIMv2Model): The AIMv2Model instance.

    Returns:
        dict[str, Any]: Configuration dictionary in HuggingFace AIMv2 format.
    """
    if model._original_config is not None:
        return model._original_config.copy()

    enc = model.encoder
    patch_size = enc.patch_embeddings.kernel_size[0]
    hidden_size = enc.patch_embeddings.out_features
    n_patches = enc.position_embeddings[...].shape[1]
    img_size = int(n_patches**0.5) * patch_size

    return {
        "model_type": "aimv2_vision_model",
        "architectures": ["Aimv2VisionModel"],
        "hidden_size": hidden_size,
        "num_hidden_layers": enc.num_layers,
        "num_attention_heads": enc.num_heads,
        "intermediate_size": enc.mlp_dim,
        "image_size": img_size,
        "patch_size": patch_size,
        "num_channels": enc.patch_embeddings.in_features,
        "hidden_act": "silu",
        "rms_norm_eps": enc.layernorm_epsilon,
        "qkv_bias": False,
        "mlp_bias": False,
        "use_bias": True,
        "use_head": False,
        "is_native": False,
        "attention_dropout": 0.0,
        "projection_dropout": 0.0,
        "initializer_range": 0.02,
    }


def _convert_aimv2_tensor_to_hf_format(hf_key: str, tensor: Array) -> Array:
    """Convert AIMv2 tensor from Flax format to HuggingFace format.

    Args:
        hf_key (str): The HuggingFace key for the tensor.
        tensor (Array): The tensor to convert.

    Returns:
        Array: The converted tensor in HuggingFace format.
    """
    if hf_key == "embeddings.position_embedding.weight" and tensor.ndim == 3:
        return tensor.squeeze(0)
    if hf_key == "embeddings.patch_embed.weight" and tensor.ndim == 4:
        return jnp.transpose(tensor, (3, 2, 0, 1))
    if any(s in hf_key for s in (".q_proj.weight", ".k_proj.weight", ".v_proj.weight")):
        if tensor.ndim == 3:
            return jnp.transpose(tensor.reshape(tensor.shape[0], -1), (1, 0))
    elif ".out_proj.weight" in hf_key:
        if tensor.ndim == 3:
            return jnp.transpose(tensor.reshape(-1, tensor.shape[2]), (1, 0))
    elif hf_key.endswith(".weight") and tensor.ndim == 2:
        return jnp.transpose(tensor, (1, 0))
    return tensor


def save_pretrained(model: "AIMv2Model", save_directory: str) -> None:
    """Save the AIMv2 model weights and config in HuggingFace format.

    Args:
        model (AIMv2Model): The AIMv2Model instance to save.
        save_directory (str): Directory path where the model will be saved.
    """
    enc = model.encoder

    _SPECIAL_MAPPINGS = {
        "encoder.patch_embeddings.weight": "embeddings.patch_embed.weight",
        "encoder.patch_embeddings.bias": "embeddings.patch_embed.bias",
        "encoder.position_embeddings": "embeddings.position_embedding.weight",
        "encoder.ln_pre.weight": "embeddings.rms_norm.weight",
        "encoder.ln_post.weight": "rms_norm.weight",
    }
    _SPECIAL_RENAMINGS: dict[str, str] = {
        ".attn.query.": ".attention.q_proj.",
        ".attn.key.": ".attention.k_proj.",
        ".attn.value.": ".attention.v_proj.",
        ".attn.out.": ".attention.out_proj.",
        ".gate.": ".ffn.gate_proj.",
        ".up.": ".ffn.up_proj.",
        ".down.": ".ffn.down_proj.",
        ".norm1.": ".rms_norm1.",
        ".norm2.": ".rms_norm2.",
    }
    for i in range(enc.num_layers):
        _SPECIAL_RENAMINGS[f"encoder.layers_{i}."] = f"encoder.layers.{i}."

    os.makedirs(save_directory, exist_ok=True)

    config = _create_aimv2_config(model)
    with open(os.path.join(save_directory, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    _, state = nnx.split(model)
    state_dict = nnx.to_pure_dict(state)
    state_dict["encoder"] = expand_scanned_layers(state_dict["encoder"])

    tensor_state = filter_tensors(state_dict)
    hf_state: dict[str, Array] = {}
    for jimm_key, tensor in tensor_state.items():
        if "vision_position_ids" in jimm_key:
            continue
        hf_key = convert_key_to_hf_format(jimm_key, _SPECIAL_MAPPINGS, _SPECIAL_RENAMINGS)
        hf_state[hf_key] = _convert_aimv2_tensor_to_hf_format(hf_key, tensor)

    save_safetensors(hf_state, os.path.join(save_directory, "model.safetensors"))


def load_from_pretrained(
    cls,
    model_name_or_path: str,
    use_pytorch: bool = False,
    rngs: rnglib.Rngs | None = None,
    dtype: DTypeLike = jnp.float32,
    param_dtype: DTypeLike = jnp.float32,
    sharding: ShardingSpec | None = None,
    use_gradient_checkpointing: bool = False,
    attention_fn: Callable[..., Any] | None = None,
) -> "AIMv2Model":
    """Load a pretrained AIMv2 model from a local path or HuggingFace Hub.

    Args:
        cls: The AIMv2Model class.
        model_name_or_path (str): Local directory or HuggingFace model ID.
        use_pytorch (bool, optional): Load from PyTorch weights. Defaults to False.
        rngs (rnglib.Rngs | None, optional): RNG state. Defaults to nnx.Rngs(0).
        dtype (DTypeLike, optional): Computation dtype. Defaults to jnp.float32.
        param_dtype (DTypeLike, optional): Parameter dtype. Defaults to jnp.float32.
        sharding (ShardingSpec | None, optional): Sharding specification. Defaults to AIMv2Sharding().
        use_gradient_checkpointing (bool, optional): Enable gradient checkpointing. Defaults to False.
        attention_fn (Callable[..., Any] | None, optional): Custom attention function. Defaults to None.

    Returns:
        AIMv2Model: Model with pretrained weights loaded.
    """
    if rngs is None:
        rngs = nnx.Rngs(0)

    params_fstate, config_dict = load_params_and_config(model_name_or_path, use_pytorch)

    if config_dict.get("model_type") == "aimv2":
        params_fstate = {k[len("vision_model.") :]: v for k, v in params_fstate.items() if k.startswith("vision_model.")}

    parsed = cls._parse_config(config_dict)
    model = cls(
        **parsed,
        use_gradient_checkpointing=use_gradient_checkpointing,
        attention_fn=attention_fn,
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
        sharding=sharding,
    )

    apply_mapping(model, params_fstate, _get_key_and_transform_mapping(), param_dtype)
    model._original_config = config_dict.get("vision_config", config_dict)
    model.eval()
    return model
