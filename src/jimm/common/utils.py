import dataclasses
import json
import os
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from huggingface_hub import hf_hub_download
from jax.experimental import multihost_utils
from jaxtyping import Array
from safetensors.flax import load_file as load_safetensors_flax_file


@dataclasses.dataclass(unsafe_hash=True)
class MeshRules:
    batch: str | None
    embed: str | None
    mlp: str | None
    vocab: str | None
    heads: str | None
    sequence_length: str | None
    singleton: str | None
    patch_height: str | None
    patch_width: str | None
    channels: str | None
    attn_features: str | None
    attn_heads: str | None
    attn_head_dim: str | None
    attn_bias_heads: str | None
    attn_bias_head_dim: str | None
    qkv_in: str | None
    qkv_out: str | None
    attn_out_in: str | None
    attn_out_out: str | None
    mlp_up_in: str | None
    mlp_up_out: str | None
    mlp_down_in: str | None
    mlp_down_out: str | None
    layernorm_dim: str | None
    patch_conv_h: str | None
    patch_conv_w: str | None
    patch_conv_c: str | None
    patch_conv_out: str | None
    cls_token_batch: str | None
    cls_token_seq: str | None
    cls_token_hidden: str | None
    probe_token_batch: str | None
    probe_token_seq: str | None
    probe_token_hidden: str | None
    map_attn_in: str | None
    map_attn_out: str | None
    map_mlp_in: str | None
    map_mlp_out: str | None
    token_embed_vocab: str | None
    token_embed_hidden: str | None
    pos_embed_seq: str | None
    pos_embed_hidden: str | None
    visual_proj_in: str | None
    visual_proj_out: str | None
    text_proj_in: str | None
    text_proj_out: str | None
    logit_scale_dim: str | None
    classifier_in: str | None
    classifier_out: str | None

    def __call__(self, *keys: str) -> tuple[str | None, ...]:
        return tuple(getattr(self, key) for key in keys)


DEFAULT_SHARDING = MeshRules(
    batch="data",
    embed=None,
    mlp=None,
    vocab="fsdp",
    heads=None,
    sequence_length=None,
    singleton=None,
    patch_height=None,
    patch_width=None,
    channels=None,
    attn_features=None,
    attn_heads=None,
    attn_head_dim=None,
    attn_bias_heads=None,
    attn_bias_head_dim=None,
    qkv_in=None,
    qkv_out=None,
    attn_out_in=None,
    attn_out_out=None,
    mlp_up_in=None,
    mlp_up_out=None,
    mlp_down_in=None,
    mlp_down_out=None,
    layernorm_dim="fsdp",
    patch_conv_h=None,
    patch_conv_w=None,
    patch_conv_c=None,
    patch_conv_out=None,
    cls_token_batch=None,
    cls_token_seq=None,
    cls_token_hidden=None,
    probe_token_batch=None,
    probe_token_seq=None,
    probe_token_hidden=None,
    map_attn_in=None,
    map_attn_out=None,
    map_mlp_in=None,
    map_mlp_out=None,
    token_embed_vocab="fsdp",
    token_embed_hidden=None,
    pos_embed_seq=None,
    pos_embed_hidden="fsdp",
    visual_proj_in=None,
    visual_proj_out="fsdp",
    text_proj_in="fsdp",
    text_proj_out=None,
    logit_scale_dim=None,
    classifier_in=None,
    classifier_out="fsdp",
)


def filter_tensors(state_dict: Dict) -> Dict[str, Array]:
    """Filter valid tensors from model state.

    Args:
        state_dict: Model state dictionary

    Returns:
        Filtered tensor dictionary
    """
    filtered = {}

    def process_item(key_name: str, value, prefix: str = ""):
        full_key = f"{prefix}.{key_name}" if prefix else key_name
        if "attn_mask" in full_key or "rngs" in full_key:
            return
        if isinstance(value, jax.Array):
            if "key<" not in str(value.dtype) and "prng" not in str(value.dtype).lower():
                if not value.is_fully_addressable:
                    value = multihost_utils.process_allgather(value, tiled=True)
                filtered[full_key] = jax.device_get(value)
        elif isinstance(value, dict):
            for nested_key, nested_value in value.items():
                process_item(nested_key, nested_value, full_key)

    for key, value in state_dict.items():
        process_item(key, value)
    return filtered


def convert_tensor_to_hf_format(hf_key: str, tensor: Array) -> Array:
    """Convert JIMM tensor to HuggingFace format.

    Args:
        hf_key: HuggingFace parameter key
        tensor: JIMM tensor

    Returns:
        HuggingFace format tensor
    """
    if ".self_attn.q_proj.weight" in hf_key or ".self_attn.k_proj.weight" in hf_key or ".self_attn.v_proj.weight" in hf_key:
        if tensor.ndim == 3:
            return tensor.reshape(-1, tensor.shape[0]).T
    elif ".self_attn.q_proj.bias" in hf_key or ".self_attn.k_proj.bias" in hf_key or ".self_attn.v_proj.bias" in hf_key:
        if tensor.ndim == 2:
            return tensor.flatten()
    elif ".self_attn.out_proj.weight" in hf_key:
        if tensor.ndim == 3:
            return tensor.reshape(tensor.shape[2], -1).T
    elif "patch_embedding.weight" in hf_key:
        if tensor.ndim == 4:
            return jnp.transpose(tensor, (3, 2, 0, 1))
    elif "class_embedding" in hf_key:
        if tensor.ndim == 3:
            return tensor.squeeze()
    elif "position_embedding.weight" in hf_key and "vision_model" in hf_key:
        if tensor.ndim == 3:
            return tensor.squeeze(0)
    elif "token_embedding.weight" in hf_key or "position_embedding.weight" in hf_key:
        return tensor
    elif hf_key.endswith(".weight") and tensor.ndim == 2:
        return tensor.T
    return tensor


def convert_key_to_hf_format(key: str, special_mappings: dict[str, str], special_renamings: dict[str, str]) -> str:
    """Convert JIMM parameter key to HuggingFace format.

    Args:
        key: JIMM parameter key

    Returns:
        HuggingFace format key
    """
    key = key.replace(".scale", ".weight")
    key = key.replace(".kernel", ".weight")
    for old, new in special_renamings.items():
        key = key.replace(old, new)
    return special_mappings.get(key, key)


def convert_state_to_hf_format(state_dict: Dict, special_mappings: dict[str, str], special_renamings: dict[str, str]) -> Dict[str, Array]:
    """Convert JIMM model state to HuggingFace format.

    Args:
        state_dict: JIMM model state dictionary

    Returns:
        HuggingFace format state dictionary
    """
    tensor_state = filter_tensors(state_dict)
    hf_state = {}

    for jimm_key, tensor in tensor_state.items():
        hf_key = convert_key_to_hf_format(jimm_key, special_mappings, special_renamings)
        hf_tensor = convert_tensor_to_hf_format(hf_key, tensor)
        hf_state[hf_key] = hf_tensor

    return hf_state


def load_params_and_config(
    model_name_or_path: str,
    use_pytorch: bool = False,
    default_config_filename: str = "config.json",
    default_pytorch_filename: str = "pytorch_model.bin",
    default_safetensors_filename: str = "model.safetensors",
) -> Tuple[Dict[str, Array], Dict[str, Any]]:
    """Loads model parameters and configuration from local directory or HuggingFace Hub.

    Args:
        model_name_or_path (str): Local directory path or HuggingFace model ID containing both weights and config.json.
        use_pytorch (bool): Whether to load from PyTorch weights. Defaults to False.
        default_config_filename (str): Config filename. Defaults to "config.json".
        default_pytorch_filename (str): PyTorch weights filename. Defaults to "pytorch_model.bin".
        default_safetensors_filename (str): Safetensors filename. Defaults to "model.safetensors".

    Returns:
        Tuple[Dict[str, Array], Dict[str, Any]]: Loaded parameters and configuration.
    """
    if os.path.isdir(model_name_or_path):
        config_file_path = os.path.join(model_name_or_path, default_config_filename)
        weights_filename = default_pytorch_filename if use_pytorch else default_safetensors_filename
        weights_file_path = os.path.join(model_name_or_path, weights_filename)
    else:
        config_file_path = hf_hub_download(repo_id=model_name_or_path, filename=default_config_filename)
        weights_filename = default_pytorch_filename if use_pytorch else default_safetensors_filename
        weights_file_path = hf_hub_download(repo_id=model_name_or_path, filename=weights_filename)

    with open(config_file_path, "r") as f:
        config = json.load(f)

    if use_pytorch:
        import torch

        state_dict = torch.load(weights_file_path, map_location="cpu")
        params_fstate = {k: jnp.array(v.numpy()) for k, v in state_dict.items()}
    else:
        params_fstate = load_safetensors_flax_file(weights_file_path)

    return params_fstate, config
