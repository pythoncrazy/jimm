import json
import os
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib
from jaxtyping import DTypeLike
from safetensors.flax import save_file as save_safetensors

from jimm.common.loading_utils import (
    apply_mapping,
    expand_scanned_layers,
    load_params_and_config,
)
from jimm.common.sharding import ShardingSpec
from jimm.common.utils import convert_state_to_hf_format

if TYPE_CHECKING:
    from jimm.models import SigLIP, SigLIPTextModel, SigLIPVisionModel


class _Transform(Enum):
    BIAS = (None, None, False)
    LINEAR = ((1, 0), None, False)
    CONV2D = ((2, 3, 1, 0), None, False)
    DEFAULT = (None, None, False)


def _preprocess_fused_qkv(params_fstate: dict[str, Any]) -> dict[str, Any]:
    """Split fused in_proj_weight/in_proj_bias keys into separate q/k/v entries.

    Args:
        params_fstate (dict[str, Any]): Raw HuggingFace parameter state dict.

    Returns:
        dict[str, Any]: Parameter dict with fused QKV split into separate keys.
    """
    processed: dict[str, Any] = {}
    for k, v in params_fstate.items():
        if k.endswith(".attention.in_proj_weight"):
            base = k[: -len(".in_proj_weight")]
            q, kw, vw = jnp.split(v, 3, axis=0)
            processed[f"{base}.q_weight"] = q
            processed[f"{base}.k_weight"] = kw
            processed[f"{base}.v_weight"] = vw
        elif k.endswith(".attention.in_proj_bias"):
            base = k[: -len(".in_proj_bias")]
            qb, kb, vb = jnp.split(v, 3, axis=0)
            processed[f"{base}.q_bias"] = qb
            processed[f"{base}.k_bias"] = kb
            processed[f"{base}.v_bias"] = vb
        else:
            processed[k] = v
    return processed


def _vision_mapping(flax_prefix: str, hf_prefix: str = "vision_model") -> dict[str, tuple[str, _Transform]]:
    """Build regex key mapping from HuggingFace to Flax format for a SigLIP vision encoder.

    Args:
        flax_prefix (str): Prefix for flax state keys (e.g. "" or "vision_model").
        hf_prefix (str): Prefix for HF keys. Defaults to "vision_model".

    Returns:
        dict[str, tuple[str, _Transform]]: Dict of {regex_pattern: (flax_key_template, Transform)}.
    """
    fp = f"{flax_prefix}." if flax_prefix else ""
    hp = f"{hf_prefix}\\." if hf_prefix else ""
    return {
        rf"{hp}embeddings\.position_embedding\.weight$": (f"{fp}encoder.position_embeddings", _Transform.DEFAULT),
        rf"{hp}embeddings\.patch_embedding\.weight$": (f"{fp}encoder.patch_embeddings.kernel", _Transform.CONV2D),
        rf"{hp}embeddings\.patch_embedding\.bias$": (f"{fp}encoder.patch_embeddings.bias", _Transform.BIAS),
        rf"{hp}post_layernorm\.weight$": (f"{fp}encoder.ln_post.scale", _Transform.DEFAULT),
        rf"{hp}post_layernorm\.bias$": (f"{fp}encoder.ln_post.bias", _Transform.BIAS),
        rf"{hp}encoder\.layers\.([0-9]+)\.self_attn\.q_proj\.weight$": (rf"{fp}encoder.layers_\1.attn.query.kernel", _Transform.LINEAR),
        rf"{hp}encoder\.layers\.([0-9]+)\.self_attn\.k_proj\.weight$": (rf"{fp}encoder.layers_\1.attn.key.kernel", _Transform.LINEAR),
        rf"{hp}encoder\.layers\.([0-9]+)\.self_attn\.v_proj\.weight$": (rf"{fp}encoder.layers_\1.attn.value.kernel", _Transform.LINEAR),
        rf"{hp}encoder\.layers\.([0-9]+)\.self_attn\.q_proj\.bias$": (rf"{fp}encoder.layers_\1.attn.query.bias", _Transform.BIAS),
        rf"{hp}encoder\.layers\.([0-9]+)\.self_attn\.k_proj\.bias$": (rf"{fp}encoder.layers_\1.attn.key.bias", _Transform.BIAS),
        rf"{hp}encoder\.layers\.([0-9]+)\.self_attn\.v_proj\.bias$": (rf"{fp}encoder.layers_\1.attn.value.bias", _Transform.BIAS),
        rf"{hp}encoder\.layers\.([0-9]+)\.self_attn\.out_proj\.weight$": (rf"{fp}encoder.layers_\1.attn.out.kernel", _Transform.LINEAR),
        rf"{hp}encoder\.layers\.([0-9]+)\.self_attn\.out_proj\.bias$": (rf"{fp}encoder.layers_\1.attn.out.bias", _Transform.BIAS),
        rf"{hp}encoder\.layers\.([0-9]+)\.layer_norm1\.weight$": (rf"{fp}encoder.layers_\1.norm1.scale", _Transform.DEFAULT),
        rf"{hp}encoder\.layers\.([0-9]+)\.layer_norm1\.bias$": (rf"{fp}encoder.layers_\1.norm1.bias", _Transform.BIAS),
        rf"{hp}encoder\.layers\.([0-9]+)\.layer_norm2\.weight$": (rf"{fp}encoder.layers_\1.norm2.scale", _Transform.DEFAULT),
        rf"{hp}encoder\.layers\.([0-9]+)\.layer_norm2\.bias$": (rf"{fp}encoder.layers_\1.norm2.bias", _Transform.BIAS),
        rf"{hp}encoder\.layers\.([0-9]+)\.mlp\.fc1\.weight$": (rf"{fp}encoder.layers_\1.mlp.layers.0.kernel", _Transform.LINEAR),
        rf"{hp}encoder\.layers\.([0-9]+)\.mlp\.fc1\.bias$": (rf"{fp}encoder.layers_\1.mlp.layers.0.bias", _Transform.BIAS),
        rf"{hp}encoder\.layers\.([0-9]+)\.mlp\.fc2\.weight$": (rf"{fp}encoder.layers_\1.mlp.layers.3.kernel", _Transform.LINEAR),
        rf"{hp}encoder\.layers\.([0-9]+)\.mlp\.fc2\.bias$": (rf"{fp}encoder.layers_\1.mlp.layers.3.bias", _Transform.BIAS),
        rf"{hp}head\.probe$": (f"{fp}encoder.map_head.probe", _Transform.DEFAULT),
        rf"{hp}head\.layernorm\.weight$": (f"{fp}encoder.map_head.layernorm.scale", _Transform.DEFAULT),
        rf"{hp}head\.layernorm\.bias$": (f"{fp}encoder.map_head.layernorm.bias", _Transform.BIAS),
        rf"{hp}head\.attention\.q_weight$": (f"{fp}encoder.map_head.attn.query.kernel", _Transform.LINEAR),
        rf"{hp}head\.attention\.k_weight$": (f"{fp}encoder.map_head.attn.key.kernel", _Transform.LINEAR),
        rf"{hp}head\.attention\.v_weight$": (f"{fp}encoder.map_head.attn.value.kernel", _Transform.LINEAR),
        rf"{hp}head\.attention\.q_bias$": (f"{fp}encoder.map_head.attn.query.bias", _Transform.BIAS),
        rf"{hp}head\.attention\.k_bias$": (f"{fp}encoder.map_head.attn.key.bias", _Transform.BIAS),
        rf"{hp}head\.attention\.v_bias$": (f"{fp}encoder.map_head.attn.value.bias", _Transform.BIAS),
        rf"{hp}head\.attention\.out_proj\.weight$": (f"{fp}encoder.map_head.attn.out.kernel", _Transform.LINEAR),
        rf"{hp}head\.attention\.out_proj\.bias$": (f"{fp}encoder.map_head.attn.out.bias", _Transform.BIAS),
        rf"{hp}head\.mlp\.fc1\.weight$": (f"{fp}encoder.map_head.mlp.layers.0.kernel", _Transform.LINEAR),
        rf"{hp}head\.mlp\.fc1\.bias$": (f"{fp}encoder.map_head.mlp.layers.0.bias", _Transform.BIAS),
        rf"{hp}head\.mlp\.fc2\.weight$": (f"{fp}encoder.map_head.mlp.layers.2.kernel", _Transform.LINEAR),
        rf"{hp}head\.mlp\.fc2\.bias$": (f"{fp}encoder.map_head.mlp.layers.2.bias", _Transform.BIAS),
    }


def _text_mapping(flax_prefix: str, hf_prefix: str = "text_model") -> dict[str, tuple[str, _Transform]]:
    """Build regex key mapping from HuggingFace to Flax format for a SigLIP text encoder.

    Args:
        flax_prefix (str): Prefix for flax state keys (e.g. "" or "text_model").
        hf_prefix (str): Prefix for HF keys. Defaults to "text_model".

    Returns:
        dict[str, tuple[str, _Transform]]: Dict of {regex_pattern: (flax_key_template, Transform)}.
    """
    fp = f"{flax_prefix}." if flax_prefix else ""
    hp = f"{hf_prefix}\\." if hf_prefix else ""
    return {
        rf"{hp}embeddings\.token_embedding\.weight$": (f"{fp}token_embedding.embedding", _Transform.DEFAULT),
        rf"{hp}embeddings\.position_embedding\.weight$": (f"{fp}positional_embedding", _Transform.DEFAULT),
        rf"{hp}final_layer_norm\.weight$": (f"{fp}ln_final.scale", _Transform.DEFAULT),
        rf"{hp}final_layer_norm\.bias$": (f"{fp}ln_final.bias", _Transform.BIAS),
        rf"{hp}head\.weight$": (f"{fp}text_projection.kernel", _Transform.LINEAR),
        rf"{hp}head\.bias$": (f"{fp}text_projection.bias", _Transform.BIAS),
        rf"{hp}encoder\.layers\.([0-9]+)\.self_attn\.q_proj\.weight$": (rf"{fp}transformer.layers_\1.attn.query.kernel", _Transform.LINEAR),
        rf"{hp}encoder\.layers\.([0-9]+)\.self_attn\.k_proj\.weight$": (rf"{fp}transformer.layers_\1.attn.key.kernel", _Transform.LINEAR),
        rf"{hp}encoder\.layers\.([0-9]+)\.self_attn\.v_proj\.weight$": (rf"{fp}transformer.layers_\1.attn.value.kernel", _Transform.LINEAR),
        rf"{hp}encoder\.layers\.([0-9]+)\.self_attn\.q_proj\.bias$": (rf"{fp}transformer.layers_\1.attn.query.bias", _Transform.BIAS),
        rf"{hp}encoder\.layers\.([0-9]+)\.self_attn\.k_proj\.bias$": (rf"{fp}transformer.layers_\1.attn.key.bias", _Transform.BIAS),
        rf"{hp}encoder\.layers\.([0-9]+)\.self_attn\.v_proj\.bias$": (rf"{fp}transformer.layers_\1.attn.value.bias", _Transform.BIAS),
        rf"{hp}encoder\.layers\.([0-9]+)\.self_attn\.out_proj\.weight$": (rf"{fp}transformer.layers_\1.attn.out.kernel", _Transform.LINEAR),
        rf"{hp}encoder\.layers\.([0-9]+)\.self_attn\.out_proj\.bias$": (rf"{fp}transformer.layers_\1.attn.out.bias", _Transform.BIAS),
        rf"{hp}encoder\.layers\.([0-9]+)\.layer_norm1\.weight$": (rf"{fp}transformer.layers_\1.norm1.scale", _Transform.DEFAULT),
        rf"{hp}encoder\.layers\.([0-9]+)\.layer_norm1\.bias$": (rf"{fp}transformer.layers_\1.norm1.bias", _Transform.BIAS),
        rf"{hp}encoder\.layers\.([0-9]+)\.layer_norm2\.weight$": (rf"{fp}transformer.layers_\1.norm2.scale", _Transform.DEFAULT),
        rf"{hp}encoder\.layers\.([0-9]+)\.layer_norm2\.bias$": (rf"{fp}transformer.layers_\1.norm2.bias", _Transform.BIAS),
        rf"{hp}encoder\.layers\.([0-9]+)\.mlp\.fc1\.weight$": (rf"{fp}transformer.layers_\1.mlp.layers.0.kernel", _Transform.LINEAR),
        rf"{hp}encoder\.layers\.([0-9]+)\.mlp\.fc1\.bias$": (rf"{fp}transformer.layers_\1.mlp.layers.0.bias", _Transform.BIAS),
        rf"{hp}encoder\.layers\.([0-9]+)\.mlp\.fc2\.weight$": (rf"{fp}transformer.layers_\1.mlp.layers.3.kernel", _Transform.LINEAR),
        rf"{hp}encoder\.layers\.([0-9]+)\.mlp\.fc2\.bias$": (rf"{fp}transformer.layers_\1.mlp.layers.3.bias", _Transform.BIAS),
    }


def _pack_map_head_attention(attn: dict[str, Any]) -> tuple[Any, Any]:
    """Pack separate MAP head q/k/v projections into fused HF tensors.

    Args:
        attn (dict[str, Any]): Attention sub-dict from the MAP head state, containing
            ``query``, ``key``, and ``value`` entries each with ``kernel`` and ``bias`` arrays.

    Returns:
        tuple[Any, Any]: ``(in_proj_weight, in_proj_bias)`` fused tensors in HuggingFace format.
    """
    q_weight = attn["query"]["kernel"]
    k_weight = attn["key"]["kernel"]
    v_weight = attn["value"]["kernel"]

    q_weight_hf = q_weight.reshape(q_weight.shape[0], -1).T
    k_weight_hf = k_weight.reshape(k_weight.shape[0], -1).T
    v_weight_hf = v_weight.reshape(v_weight.shape[0], -1).T
    in_proj_weight = jnp.concatenate([q_weight_hf, k_weight_hf, v_weight_hf], axis=0)

    q_bias_hf = attn["query"]["bias"].flatten()
    k_bias_hf = attn["key"]["bias"].flatten()
    v_bias_hf = attn["value"]["bias"].flatten()
    in_proj_bias = jnp.concatenate([q_bias_hf, k_bias_hf, v_bias_hf], axis=0)

    return in_proj_weight, in_proj_bias


def _rewrite_map_head_for_hf(maphead: dict[str, Any]) -> None:
    """Convert MAP head attention weights to the fused HF layout in-place.

    Merges separate ``query``, ``key``, ``value`` kernel/bias entries into a single
    ``in_proj_weight`` / ``in_proj_bias`` and moves the output projection to
    ``self_attn.out_proj``.

    Args:
        maphead (dict[str, Any]): MAP head state sub-dict, modified in place.

    Returns:
        None
    """
    if "attn" not in maphead:
        return

    attn = maphead["attn"]
    if all(k in attn for k in ["query", "key", "value"]):
        in_proj_weight, in_proj_bias = _pack_map_head_attention(attn)
        del attn["query"], attn["key"], attn["value"]
        attn["in_proj_weight"] = in_proj_weight
        attn["in_proj_bias"] = in_proj_bias

    if "out" in attn:
        out_kernel = attn["out"]["kernel"]
        out_proj_weight = out_kernel.reshape(-1, out_kernel.shape[-1])
        maphead.setdefault("self_attn", {})
        maphead["self_attn"]["out_proj"] = {"weight": out_proj_weight, "bias": attn["out"]["bias"]}
        del attn["out"]


def _create_config(model: "SigLIP") -> dict[str, Any]:
    """Create HuggingFace config dictionary from SigLIP model.

    Args:
        model (SigLIP): SigLIP model instance.

    Returns:
        dict[str, Any]: Configuration in HuggingFace format.
    """
    n_patches = model.vision_model.encoder.position_embeddings[...].shape[1]
    img_size = int(n_patches**0.5) * model.vision_patch_size
    return {
        "model_type": "siglip",
        "text_config": {
            "hidden_size": model.text_hidden_size,
            "num_attention_heads": model.num_text_heads,
            "num_hidden_layers": model.num_text_layers,
            "max_position_embeddings": model.context_length,
            "vocab_size": model.vocab_size,
        },
        "vision_config": {
            "hidden_size": model.vision_hidden_size,
            "num_attention_heads": model.vision_hidden_size // 64,
            "num_hidden_layers": model.vision_layers,
            "image_size": img_size,
            "patch_size": model.vision_patch_size,
        },
    }


def save_pretrained(model: "SigLIP", save_directory: str) -> None:
    """Save SigLIP model in HuggingFace format.

    Args:
        model (SigLIP): SigLIP model to save.
        save_directory (str): Output directory.
    """
    _SPECIAL_MAPPINGS = {
        "text_model.ln_final.weight": "text_model.final_layer_norm.weight",
        "text_model.ln_final.bias": "text_model.final_layer_norm.bias",
        "vision_model.encoder.ln_post.weight": "vision_model.post_layernorm.weight",
        "vision_model.encoder.ln_post.bias": "vision_model.post_layernorm.bias",
        "vision_model.encoder.position_embeddings": "vision_model.embeddings.position_embedding.weight",
        "vision_model.encoder.patch_embeddings.weight": "vision_model.embeddings.patch_embedding.weight",
        "vision_model.encoder.patch_embeddings.bias": "vision_model.embeddings.patch_embedding.bias",
        "text_model.positional_embedding": "text_model.embeddings.position_embedding.weight",
        "text_model.token_embedding.embedding": "text_model.embeddings.token_embedding.weight",
        "text_model.text_projection.weight": "text_model.head.weight",
        "text_model.text_projection.bias": "text_model.head.bias",
        "vision_model.encoder.map_head.probe": "vision_model.head.probe",
        "vision_model.encoder.map_head.layernorm.weight": "vision_model.head.layernorm.weight",
        "vision_model.encoder.map_head.layernorm.bias": "vision_model.head.layernorm.bias",
        "vision_model.encoder.map_head.mlp.fc1.weight": "vision_model.head.mlp.fc1.weight",
        "vision_model.encoder.map_head.mlp.fc1.bias": "vision_model.head.mlp.fc1.bias",
        "vision_model.encoder.map_head.mlp.layers.2.weight": "vision_model.head.mlp.fc2.weight",
        "vision_model.encoder.map_head.mlp.layers.2.bias": "vision_model.head.mlp.fc2.bias",
        "vision_model.encoder.map_head.attn.in_proj_weight": "vision_model.head.attention.in_proj_weight",
        "vision_model.encoder.map_head.attn.in_proj_bias": "vision_model.head.attention.in_proj_bias",
        "vision_model.encoder.map_head.self_attn.out_proj.weight": "vision_model.head.attention.out_proj.weight",
        "vision_model.encoder.map_head.self_attn.out_proj.bias": "vision_model.head.attention.out_proj.bias",
    }
    _SPECIAL_RENAMINGS: dict[str, str] = {
        "text_model.transformer.layers": "text_model.encoder.layers",
        "vision_model.encoder.layers": "vision_model.encoder.layers",
        ".attn.query.": ".self_attn.q_proj.",
        ".attn.key.": ".self_attn.k_proj.",
        ".attn.value.": ".self_attn.v_proj.",
        ".attn.out.": ".self_attn.out_proj.",
        ".mlp.layers.0.": ".mlp.fc1.",
        ".mlp.layers.3.": ".mlp.fc2.",
        ".norm1.": ".layer_norm1.",
        ".norm2.": ".layer_norm2.",
    }
    for i in range(100):
        _SPECIAL_RENAMINGS[f"layers_{i}."] = f"layers.{i}."

    os.makedirs(save_directory, exist_ok=True)

    config = model._original_config.copy() if model._original_config else _create_config(model)
    with open(os.path.join(save_directory, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    _, state = nnx.split(model)
    state_dict = expand_scanned_layers(nnx.to_pure_dict(state))

    if "vision_model" in state_dict and "map_head" in state_dict["vision_model"]["encoder"]:
        _rewrite_map_head_for_hf(state_dict["vision_model"]["encoder"]["map_head"])

    hf_state = convert_state_to_hf_format(state_dict, _SPECIAL_MAPPINGS, _SPECIAL_RENAMINGS)

    for key in ["logit_scale", "logit_bias"]:
        if key in hf_state and hf_state[key].ndim == 0:
            hf_state[key] = jnp.expand_dims(hf_state[key], 0)

    hf_state.pop("vision_model.encoder.vision_position_ids", None)
    save_safetensors(hf_state, os.path.join(save_directory, "model.safetensors"))


def save_vision_pretrained(model: "SigLIPVisionModel", save_directory: str) -> None:
    """Save SigLIP vision model in HuggingFace format.

    Args:
        model (SigLIPVisionModel): Model to save.
        save_directory (str): Output directory.
    """
    _SPECIAL_MAPPINGS = {
        "encoder.ln_post.weight": "post_layernorm.weight",
        "encoder.ln_post.bias": "post_layernorm.bias",
        "encoder.position_embeddings": "embeddings.position_embedding.weight",
        "encoder.patch_embeddings.weight": "embeddings.patch_embedding.weight",
        "encoder.patch_embeddings.bias": "embeddings.patch_embedding.bias",
        "encoder.map_head.probe": "head.probe",
        "encoder.map_head.layernorm.weight": "head.layernorm.weight",
        "encoder.map_head.layernorm.bias": "head.layernorm.bias",
        "encoder.map_head.mlp.fc1.weight": "head.mlp.fc1.weight",
        "encoder.map_head.mlp.fc1.bias": "head.mlp.fc1.bias",
        "encoder.map_head.mlp.layers.2.weight": "head.mlp.fc2.weight",
        "encoder.map_head.mlp.layers.2.bias": "head.mlp.fc2.bias",
        "encoder.map_head.attn.in_proj_weight": "head.attention.in_proj_weight",
        "encoder.map_head.attn.in_proj_bias": "head.attention.in_proj_bias",
        "encoder.map_head.self_attn.out_proj.weight": "head.attention.out_proj.weight",
        "encoder.map_head.self_attn.out_proj.bias": "head.attention.out_proj.bias",
    }
    _SPECIAL_RENAMINGS: dict[str, str] = {
        ".attn.query.": ".self_attn.q_proj.",
        ".attn.key.": ".self_attn.k_proj.",
        ".attn.value.": ".self_attn.v_proj.",
        ".attn.out.": ".self_attn.out_proj.",
        ".mlp.layers.0.": ".mlp.fc1.",
        ".mlp.layers.3.": ".mlp.fc2.",
        ".norm1.": ".layer_norm1.",
        ".norm2.": ".layer_norm2.",
    }
    for i in range(100):
        _SPECIAL_RENAMINGS[f".layers_{i}."] = f".layers.{i}."

    os.makedirs(save_directory, exist_ok=True)

    n_patches = model.encoder.position_embeddings[...].shape[1]
    img_size = int(n_patches**0.5) * model.vision_patch_size
    vision_config = {
        "hidden_size": model.vision_hidden_size,
        "image_size": img_size,
        "intermediate_size": model.vision_hidden_size * 4,
        "num_attention_heads": model.vision_hidden_size // 64,
        "num_hidden_layers": model.vision_layers,
        "patch_size": model.vision_patch_size,
    }
    config = {"vision_config": vision_config, "model_type": "siglip"}

    with open(os.path.join(save_directory, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    _, state = nnx.split(model)
    state_dict = expand_scanned_layers(nnx.to_pure_dict(state))

    if "encoder" in state_dict and "map_head" in state_dict["encoder"]:
        _rewrite_map_head_for_hf(state_dict["encoder"]["map_head"])

    prefixed_renamings: dict[str, str] = {}
    for k, v in _SPECIAL_RENAMINGS.items():
        if k.startswith("."):
            prefixed_renamings[k] = v
        else:
            prefixed_renamings["vision_model." + k] = "vision_model." + v

    hf_state = convert_state_to_hf_format(
        {"vision_model": state_dict},
        {"vision_model." + k: "vision_model." + v for k, v in _SPECIAL_MAPPINGS.items()},
        prefixed_renamings,
    )
    hf_state.pop("vision_model.encoder.vision_position_ids", None)
    save_safetensors(hf_state, os.path.join(save_directory, "model.safetensors"))


def save_text_pretrained(model: "SigLIPTextModel", save_directory: str) -> None:
    """Save SigLIP text model in HuggingFace format.

    Args:
        model (SigLIPTextModel): Model to save.
        save_directory (str): Output directory.
    """
    _SPECIAL_MAPPINGS = {
        "ln_final.weight": "final_layer_norm.weight",
        "ln_final.bias": "final_layer_norm.bias",
        "positional_embedding": "embeddings.position_embedding.weight",
        "token_embedding.embedding": "embeddings.token_embedding.weight",
        "text_projection.weight": "head.weight",
        "text_projection.bias": "head.bias",
    }
    _SPECIAL_RENAMINGS: dict[str, str] = {
        "transformer.layers": "encoder.layers",
        ".attn.query.": ".self_attn.q_proj.",
        ".attn.key.": ".self_attn.k_proj.",
        ".attn.value.": ".self_attn.v_proj.",
        ".attn.out.": ".self_attn.out_proj.",
        ".mlp.layers.0.": ".mlp.fc1.",
        ".mlp.layers.3.": ".mlp.fc2.",
        ".norm1.": ".layer_norm1.",
        ".norm2.": ".layer_norm2.",
    }
    for i in range(100):
        _SPECIAL_RENAMINGS[f".layers_{i}."] = f".layers.{i}."

    os.makedirs(save_directory, exist_ok=True)

    text_config = {
        "hidden_size": model.text_hidden_size,
        "intermediate_size": model.text_hidden_size * 4,
        "num_attention_heads": model.num_text_heads,
        "num_hidden_layers": model.num_text_layers,
        "max_position_embeddings": model.context_length,
        "vocab_size": model.vocab_size,
    }
    config = {"text_config": text_config, "model_type": "siglip"}

    with open(os.path.join(save_directory, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    _, state = nnx.split(model)
    state_dict = expand_scanned_layers(nnx.to_pure_dict(state))

    prefixed_renamings: dict[str, str] = {}
    for k, v in _SPECIAL_RENAMINGS.items():
        if k.startswith("."):
            prefixed_renamings[k] = v
        else:
            prefixed_renamings["text_model." + k] = "text_model." + v

    hf_state = convert_state_to_hf_format(
        {"text_model": state_dict},
        {"text_model." + k: "text_model." + v for k, v in _SPECIAL_MAPPINGS.items()},
        prefixed_renamings,
    )
    save_safetensors(hf_state, os.path.join(save_directory, "model.safetensors"))


def load_text_from_pretrained(
    cls,
    model_name_or_path: str,
    use_pytorch: bool,
    rngs: rnglib.Rngs | None,
    dtype: DTypeLike,
    param_dtype: DTypeLike,
    sharding: ShardingSpec,
    use_gradient_checkpointing: bool,
    attention_fn: Callable[..., Any] | None = None,
) -> "SigLIPTextModel":
    """Load pretrained SigLIP text model.

    Args:
        cls: SigLIPTextModel class.
        model_name_or_path (str): Model path or ID.
        use_pytorch (bool): Load from PyTorch.
        rngs (rnglib.Rngs | None): RNG state. If None, initializes to nnx.Rngs(0).
        dtype (DTypeLike): Computation dtype.
        param_dtype (DTypeLike): Parameter dtype.
        sharding (ShardingSpec): Sharding specification for parameters.
        use_gradient_checkpointing (bool): Enable gradient checkpointing.
        attention_fn (Callable[..., Any] | None): Custom attention function (e.g. jimm.tokamax_attention). Defaults to None.

    Returns:
        SigLIPTextModel: Loaded model.
    """
    if rngs is None:
        rngs = nnx.Rngs(0)
    params_fstate, config_dict = load_params_and_config(model_name_or_path, use_pytorch)

    context_length = params_fstate["text_model.embeddings.position_embedding.weight"].shape[0]
    vocab_size = params_fstate["text_model.embeddings.token_embedding.weight"].shape[0]
    text_hidden_size = params_fstate["text_model.embeddings.token_embedding.weight"].shape[1]
    text_num_layers = max(int(k.split(".")[3]) + 1 for k in params_fstate if k.startswith("text_model.encoder.layers.") and k.endswith(".self_attn.q_proj.weight"))

    model = cls(
        context_length=context_length,
        vocab_size=vocab_size,
        text_hidden_size=text_hidden_size,
        num_text_heads=text_hidden_size // 64,
        num_text_layers=text_num_layers,
        use_gradient_checkpointing=use_gradient_checkpointing,
        attention_fn=attention_fn,
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
        sharding=sharding,
    )

    mapping = _text_mapping(flax_prefix="", hf_prefix="text_model")
    m = apply_mapping(model, params_fstate, mapping, param_dtype)
    m.eval()
    m._original_config = config_dict
    return m


def load_vision_from_pretrained(
    cls,
    model_name_or_path: str,
    use_pytorch: bool,
    rngs: rnglib.Rngs | None,
    dtype: DTypeLike,
    param_dtype: DTypeLike,
    sharding: ShardingSpec,
    use_gradient_checkpointing: bool,
    attention_fn: Callable[..., Any] | None = None,
) -> "SigLIPVisionModel":
    """Load pretrained SigLIP vision model.

    Args:
        cls: SigLIPVisionModel class.
        model_name_or_path (str): Model path or ID.
        use_pytorch (bool): Load from PyTorch.
        rngs (rnglib.Rngs | None): RNG state. If None, initializes to nnx.Rngs(0).
        dtype (DTypeLike): Computation dtype.
        param_dtype (DTypeLike): Parameter dtype.
        sharding (ShardingSpec): Sharding specification for parameters.
        use_gradient_checkpointing (bool): Enable gradient checkpointing.
        attention_fn (Callable[..., Any] | None): Custom attention function (e.g. jimm.tokamax_attention). Defaults to None.

    Returns:
        SigLIPVisionModel: Loaded model.
    """
    if rngs is None:
        rngs = nnx.Rngs(0)
    params_fstate, config_dict = load_params_and_config(model_name_or_path, use_pytorch)
    params_fstate = _preprocess_fused_qkv(params_fstate)

    vision_patch_size = params_fstate["vision_model.embeddings.patch_embedding.weight"].shape[3]
    vision_width = params_fstate["vision_model.embeddings.patch_embedding.bias"].shape[0]
    vision_num_layers = max(int(k.split(".")[3]) + 1 for k in params_fstate if k.startswith("vision_model.encoder.layers.") and k.endswith(".mlp.fc2.bias"))
    image_resolution = config_dict["vision_config"].get("image_size") or (int(params_fstate["vision_model.embeddings.position_embedding.weight"].shape[0] ** 0.5) * vision_patch_size)

    model = cls(
        image_resolution=image_resolution,
        vision_layers=vision_num_layers,
        vision_hidden_size=vision_width,
        vision_patch_size=vision_patch_size,
        use_gradient_checkpointing=use_gradient_checkpointing,
        attention_fn=attention_fn,
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
        sharding=sharding,
    )

    mapping = _vision_mapping(flax_prefix="", hf_prefix="vision_model")
    m = apply_mapping(model, params_fstate, mapping, param_dtype)
    m.eval()
    m._original_config = config_dict
    return m


def load_from_pretrained(
    cls,
    model_name_or_path: str,
    use_pytorch: bool,
    rngs: rnglib.Rngs | None,
    dtype: DTypeLike,
    param_dtype: DTypeLike,
    sharding: ShardingSpec,
    use_gradient_checkpointing: bool,
    attention_fn: Callable[..., Any] | None = None,
) -> "SigLIP":
    """Load pretrained SigLIP model.

    Args:
        cls: SigLIP class.
        model_name_or_path (str): Model path or ID.
        use_pytorch (bool): Load from PyTorch.
        rngs (rnglib.Rngs | None): RNG state. If None, initializes to nnx.Rngs(0).
        dtype (DTypeLike): Computation dtype.
        param_dtype (DTypeLike): Parameter dtype.
        sharding (ShardingSpec): Sharding specification for parameters.
        use_gradient_checkpointing (bool): Enable gradient checkpointing.
        attention_fn (Callable[..., Any] | None): Custom attention function (e.g. jimm.tokamax_attention). Defaults to None.

    Returns:
        SigLIP: Loaded model.
    """
    if rngs is None:
        rngs = nnx.Rngs(0)
    params_fstate, config_dict = load_params_and_config(model_name_or_path, use_pytorch)
    params_fstate = _preprocess_fused_qkv(params_fstate)

    vision_patch_size = params_fstate["vision_model.embeddings.patch_embedding.weight"].shape[3]
    vision_width = params_fstate["vision_model.embeddings.patch_embedding.bias"].shape[0]
    vision_num_layers = max(int(k.split(".")[3]) + 1 for k in params_fstate if k.startswith("vision_model.encoder.layers.") and k.endswith(".mlp.fc2.bias"))
    context_length = params_fstate["text_model.embeddings.position_embedding.weight"].shape[0]
    vocab_size = params_fstate["text_model.embeddings.token_embedding.weight"].shape[0]
    text_hidden_size = params_fstate["text_model.embeddings.token_embedding.weight"].shape[1]
    text_num_layers = max(int(k.split(".")[3]) + 1 for k in params_fstate if k.startswith("text_model.encoder.layers.") and k.endswith(".self_attn.q_proj.weight"))
    image_resolution = config_dict["vision_config"].get("image_size") or (int(params_fstate["vision_model.embeddings.position_embedding.weight"].shape[0] ** 0.5) * vision_patch_size)

    model = cls(
        image_resolution=image_resolution,
        vision_layers=vision_num_layers,
        vision_hidden_size=vision_width,
        vision_patch_size=vision_patch_size,
        context_length=context_length,
        vocab_size=vocab_size,
        text_hidden_size=text_hidden_size,
        num_text_heads=text_hidden_size // 64,
        num_text_layers=text_num_layers,
        use_gradient_checkpointing=use_gradient_checkpointing,
        attention_fn=attention_fn,
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
        sharding=sharding,
    )

    mapping = {
        **_vision_mapping(flax_prefix="vision_model", hf_prefix="vision_model"),
        **_text_mapping(flax_prefix="text_model", hf_prefix="text_model"),
        r"logit_scale$": ("logit_scale", _Transform.DEFAULT),
        r"logit_bias$": ("logit_bias", _Transform.DEFAULT),
    }
    m = apply_mapping(model, params_fstate, mapping, param_dtype)
    m.eval()
    m._original_config = config_dict
    return m
