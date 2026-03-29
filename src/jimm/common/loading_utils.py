import json
import os
import re
from math import prod
from typing import Any, Dict, Tuple, TypeVar

import jax
import jax.numpy as jnp
from flax import nnx
from huggingface_hub import hf_hub_download
from jaxtyping import Array, DTypeLike
from safetensors.flax import load_file as load_safetensors_flax_file

_M = TypeVar("_M", bound=nnx.Module)


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


def stoi(k: str) -> int | str:
    """Convert a string to int if numeric, else return as-is.

    Args:
        k (str): Key to convert.

    Returns:
        int | str: Integer if numeric string, else original string.
    """
    try:
        return int(k)
    except ValueError:
        return k


def map_to_bonsai_key(
    mapping: dict[str, tuple[str, Any]],
    key: str,
) -> tuple[str | None, Any]:
    """Match a flat HF key against regex patterns and return the flax target key.

    Args:
        mapping (dict[str, tuple[str, Any]]): Dict of {regex_pattern: (flax_key_template, transform)}.
        key (str): HuggingFace parameter key to match.

    Returns:
        tuple[str | None, Any]: (flax_key, transform) if matched, else (None, None).
    """
    for pattern, (target, transform) in mapping.items():
        if re.fullmatch(pattern, key):
            return re.sub(pattern, target, key), transform
    return None, None


def to_scan_batched_keys(keys: tuple) -> tuple[tuple | None, int | None]:
    """Convert a per-layer flat state key to its scan-batched equivalent.

    With nnx.scan/vmap, transformer layers are stored as a single batched module
    under a "layers" key rather than separate "layers_N" keys. This converts
    keys like ("encoder", "layers_3", "attn", "query", "kernel") to
    ("encoder", "layers", "attn", "query", "kernel") and returns the layer index.

    Args:
        keys (tuple): Flat state key tuple potentially containing a "layers_N" component.

    Returns:
        tuple[tuple | None, int | None]: (batched_keys, layer_idx) if a layers_N component
            is found, else (None, None).
    """
    new_keys = list(keys)
    layer_idx = None
    for i, k in enumerate(new_keys):
        if isinstance(k, str) and k.startswith("layers_"):
            try:
                layer_idx = int(k[7:])
                new_keys[i] = "layers"
                break
            except ValueError:
                pass
    if layer_idx is None:
        return None, None
    return tuple(new_keys), layer_idx


def _reshape_if_compatible(tensor: Array, target_shape: tuple[int, ...], hf_key: str, bonsai_keys: tuple[Any, ...]) -> Array:
    """Reshape tensor only when element counts match, else raise a clear error."""
    if tensor.shape == target_shape:
        return tensor

    if tensor.size != prod(target_shape):
        bonsai_key = ".".join(str(key) for key in bonsai_keys)
        raise ValueError(f"Shape mismatch for {hf_key} -> {bonsai_key}: got {tensor.shape}, expected {target_shape}")

    return tensor.reshape(target_shape)


def apply_mapping(
    model: _M,
    params_fstate: dict[str, Any],
    mapping: dict[str, tuple[str, Any]],
    param_dtype: DTypeLike,
) -> _M:
    """Apply regex-based HF parameter mappings to a model in-place."""
    flat_state = dict(nnx.to_flat_state(nnx.state(model, nnx.Param)))
    layer_accum: dict[tuple, dict[int, Any]] = {}

    for hf_key, tensor in params_fstate.items():
        bonsai_key, transform = map_to_bonsai_key(mapping, hf_key)
        if bonsai_key is None:
            continue

        keys = tuple(stoi(k) for k in bonsai_key.split("."))
        permute_rule, _, _ = transform.value
        transformed = tensor.astype(param_dtype)
        if permute_rule is not None:
            transformed = jnp.transpose(transformed, permute_rule)

        if keys in flat_state:
            var = flat_state[keys]
            var[...] = _reshape_if_compatible(transformed, var[...].shape, hf_key, keys)
            continue

        batched_keys, layer_idx = to_scan_batched_keys(keys)
        if batched_keys is not None and layer_idx is not None and batched_keys in flat_state:
            layer_accum.setdefault(batched_keys, {})[layer_idx] = transformed

    for batched_keys, layers_dict in layer_accum.items():
        var = flat_state[batched_keys]
        num_layers = var[...].shape[0]
        missing = sorted(set(range(num_layers)) - set(layers_dict))
        if missing:
            bonsai_key = ".".join(str(key) for key in batched_keys)
            raise ValueError(f"Missing scanned layers for {bonsai_key}: {missing}")

        stacked = jnp.stack([layers_dict[i] for i in range(num_layers)], axis=0)
        var[...] = _reshape_if_compatible(stacked, var[...].shape, ".".join(str(key) for key in batched_keys), batched_keys)

    nnx.update(model, nnx.from_flat_state(flat_state))
    return model


def _slice_layer(d: dict, idx: int) -> dict:
    """Recursively extract index idx from batched arrays in a nested dict."""
    result = {}
    for key, value in d.items():
        if isinstance(value, dict):
            result[key] = _slice_layer(value, idx)
        elif isinstance(value, jax.Array):
            result[key] = value[idx]
        else:
            result[key] = value
    return result


def _infer_num_layers(d: dict) -> int | None:
    """Return the leading dimension of the first JAX array found in d."""
    for value in d.values():
        if isinstance(value, jax.Array):
            return int(value.shape[0])
        if isinstance(value, dict):
            n = _infer_num_layers(value)
            if n is not None:
                return n
    return None


def _is_numeric_key(key: Any) -> bool:
    """Return True when a nested state-dict key is an integer layer index."""
    return isinstance(key, int) or (isinstance(key, str) and key.isdigit())


def _should_expand_layers_dict(d: dict) -> bool:
    """Return True only for scan-batched layer dicts.

    `nnx.scan` stores stacked transformer blocks under a single ``layers`` key
    whose children are named module fields like ``attn`` and ``mlp``. In
    contrast, ``nnx.Sequential`` also uses a ``layers`` key, but its children
    are numeric submodule indices. Those sequential containers must not be
    expanded.
    """
    if not d or all(_is_numeric_key(key) for key in d):
        return False
    return _infer_num_layers(d) is not None


def expand_scanned_layers(state_dict: dict) -> dict:
    """Expand scan-batched layer parameters into per-layer entries for HF saving.

    With nnx.scan/vmap, all transformer layer parameters are stored as a single
    batched module under a "layers" key with a leading layer dimension. This
    expands them into separate "layers_N" sub-dicts compatible with the HF
    format conversion utilities.

    Args:
        state_dict (dict): Model state dict potentially containing a "layers" key
            with batched parameters.

    Returns:
        dict: State dict with "layers" expanded into "layers_0", ..., "layers_(N-1)".
    """
    result = {}
    for key, value in state_dict.items():
        if key == "layers" and isinstance(value, dict) and _should_expand_layers_dict(value):
            num_layers = _infer_num_layers(value)
            if num_layers is not None:
                for i in range(num_layers):
                    result[f"layers_{i}"] = _slice_layer(value, i)
                continue
        result[key] = expand_scanned_layers(value) if isinstance(value, dict) else value
    return result
