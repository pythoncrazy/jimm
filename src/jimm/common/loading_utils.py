import json
import os
from typing import Any, Callable, Dict, Tuple

import jax.numpy as jnp
from flax import nnx
from huggingface_hub import hf_hub_download
from jaxtyping import Array, DTypeLike
from safetensors.flax import load_file as load_safetensors_flax_file


def load_params_and_config(
    model_name_or_path: str,
    use_pytorch: bool = False,
    default_config_filename: str = "config.json",
    default_pytorch_filename: str = "pytorch_model.bin",
    default_safetensors_filename: str = "model.safetensors",
) -> Tuple[Dict[str, Array], Dict[str, Any]]:
    """Load model parameters and configuration from local directory or HuggingFace Hub.

    Args:
        model_name_or_path (str): Local directory path or HuggingFace model ID.
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


def load_and_apply_params(
    model: nnx.Module,
    params_fstate: dict[str, Any],
    mapping: dict[tuple, tuple],
    transform_fn: Callable,
    param_dtype: DTypeLike,
) -> None:
    """Load and apply parameters from HuggingFace format to Flax model.

    Args:
        model (nnx.Module): Target Flax model.
        params_fstate (dict[str, Any]): Source parameters in HuggingFace format.
        mapping (dict[tuple, tuple]): Parameter name mapping from Flax to HuggingFace.
        transform_fn (Callable): Function to transform parameters.
        param_dtype (DTypeLike): Target parameter data type.
    """
    flax_params = dict(nnx.to_flat_state(nnx.state(model, nnx.Param)))
    nonvisited = set(flax_params.keys())

    for flax_key, hf_key in mapping.items():
        hf_key_str = ".".join(hf_key)
        if hf_key_str not in params_fstate:
            continue

        nonvisited.discard(flax_key)
        value = params_fstate[hf_key_str]
        target = flax_params[flax_key]

        value = transform_fn(value, flax_key, hf_key)

        if value.shape != target[...].shape:
            raise ValueError(f"Shape mismatch for {flax_key}: expected {target[...].shape}, got {value.shape}")

        target[...] = value.astype(param_dtype)

    nnx.update(model, nnx.from_flat_state(flax_params))

    buffer_keys = {
        ("encoder", "vision_position_ids"),
        ("visual_projection",),
        ("text_position_ids",),
        ("text_model", "text_position_ids"),
        ("vision_model", "encoder", "vision_position_ids"),
    }
    unexpected = nonvisited - buffer_keys
    if unexpected:
        print(f"Warning: Parameters not loaded: {sorted(list(unexpected))}")


def build_base_vision_mapping(
    config: dict[str, Any],
    prefix: str = "",
) -> dict[tuple, tuple]:
    """Build base vision model parameter mapping (common to all models).

    Args:
        config (dict[str, Any]): Vision config dictionary.
        prefix (str): Prefix for Flax keys. Defaults to "".

    Returns:
        dict[tuple, tuple]: Parameter mapping from Flax to HuggingFace format.
    """
    pf = prefix.split(".") if prefix else []
    mapping = {
        tuple(pf + ["encoder", "position_embeddings"]): ("vision_model", "embeddings", "position_embedding", "weight"),
        tuple(pf + ["encoder", "patch_embeddings", "kernel"]): ("vision_model", "embeddings", "patch_embedding", "weight"),
        tuple(pf + ["encoder", "ln_post", "scale"]): ("vision_model", "post_layernorm", "weight"),
        tuple(pf + ["encoder", "ln_post", "bias"]): ("vision_model", "post_layernorm", "bias"),
    }

    attn_parts = [("query", "q_proj"), ("key", "k_proj"), ("value", "v_proj"), ("out", "out_proj")]
    norm_parts = [("norm1", "layer_norm1"), ("norm2", "layer_norm2")]
    mlp_parts = [(("mlp", "layers", 0), ("mlp", "fc1")), (("mlp", "layers", 3), ("mlp", "fc2"))]

    for i in range(config["num_hidden_layers"]):
        fb = tuple(pf + ["encoder", "encoder", f"layers_{i}"])
        hb = ("vision_model", "encoder", "layers", str(i))
        for flax_name, hf_name in attn_parts:
            mapping[fb + ("attn", flax_name, "kernel")] = hb + ("self_attn", hf_name, "weight")
            mapping[fb + ("attn", flax_name, "bias")] = hb + ("self_attn", hf_name, "bias")
        for flax_name, hf_name in norm_parts:
            mapping[fb + (flax_name, "scale")] = hb + (hf_name, "weight")
            mapping[fb + (flax_name, "bias")] = hb + (hf_name, "bias")
        for flax_path, hf_path in mlp_parts:
            mapping[fb + flax_path + ("kernel",)] = hb + hf_path + ("weight",)
            mapping[fb + flax_path + ("bias",)] = hb + hf_path + ("bias",)

    return mapping


def build_base_text_mapping(
    config: dict[str, Any],
    prefix: str = "",
) -> dict[tuple, tuple]:
    """Build base text model parameter mapping (common to all models).

    Args:
        config (dict[str, Any]): Text config dictionary.
        prefix (str): Prefix for Flax keys. Defaults to "".

    Returns:
        dict[tuple, tuple]: Parameter mapping from Flax to HuggingFace format.
    """
    pf = prefix.split(".") if prefix else []
    mapping = {
        tuple(pf + ["token_embedding", "embedding"]): ("text_model", "embeddings", "token_embedding", "weight"),
        tuple(pf + ["positional_embedding"]): ("text_model", "embeddings", "position_embedding", "weight"),
        tuple(pf + ["ln_final", "scale"]): ("text_model", "final_layer_norm", "weight"),
        tuple(pf + ["ln_final", "bias"]): ("text_model", "final_layer_norm", "bias"),
    }

    attn_parts = [("query", "q_proj"), ("key", "k_proj"), ("value", "v_proj"), ("out", "out_proj")]
    norm_parts = [("norm1", "layer_norm1"), ("norm2", "layer_norm2")]
    mlp_parts = [(("mlp", "layers", 0), ("mlp", "fc1")), (("mlp", "layers", 3), ("mlp", "fc2"))]

    for i in range(config["num_hidden_layers"]):
        fb = tuple(pf + ["transformer", f"layers_{i}"])
        hb = ("text_model", "encoder", "layers", str(i))
        for flax_name, hf_name in attn_parts:
            mapping[fb + ("attn", flax_name, "kernel")] = hb + ("self_attn", hf_name, "weight")
            mapping[fb + ("attn", flax_name, "bias")] = hb + ("self_attn", hf_name, "bias")
        for flax_name, hf_name in norm_parts:
            mapping[fb + (flax_name, "scale")] = hb + (hf_name, "weight")
            mapping[fb + (flax_name, "bias")] = hb + (hf_name, "bias")
        for flax_path, hf_path in mlp_parts:
            mapping[fb + flax_path + ("kernel",)] = hb + hf_path + ("weight",)
            mapping[fb + flax_path + ("bias",)] = hb + hf_path + ("bias",)

    return mapping
