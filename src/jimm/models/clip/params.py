import json
import os
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeVar

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib
from jaxtyping import DTypeLike
from safetensors.flax import save_file as save_safetensors

from jimm.common.loading_utils import (
    expand_scanned_layers,
    load_params_and_config,
    map_to_bonsai_key,
    stoi,
    to_scan_batched_keys,
)
from jimm.common.sharding import ShardingSpec
from jimm.common.utils import convert_state_to_hf_format

if TYPE_CHECKING:
    from jimm.models import CLIP, CLIPTextModel, CLIPVisionModel


_M = TypeVar("_M", bound=nnx.Module)


class _Transform(Enum):
    BIAS = (None, None, False)
    LINEAR = ((1, 0), None, False)
    CONV2D = ((2, 3, 1, 0), None, False)
    DEFAULT = (None, None, False)


def _vision_mapping(flax_prefix: str, hf_prefix: str = "vision_model") -> dict[str, tuple[str, _Transform]]:
    """Build regex key mapping from HuggingFace to Flax format for a CLIP vision encoder.

    Args:
        flax_prefix (str): Prefix for flax state keys (e.g. "" or "vision_model").
        hf_prefix (str): Prefix for HF keys. Defaults to "vision_model".

    Returns:
        dict[str, tuple[str, _Transform]]: Dict of {regex_pattern: (flax_key_template, Transform)}.
    """
    fp = f"{flax_prefix}." if flax_prefix else ""
    hp = f"{hf_prefix}\\." if hf_prefix else ""
    return {
        rf"{hp}embeddings\.class_embedding$": (f"{fp}encoder.cls_token", _Transform.DEFAULT),
        rf"{hp}embeddings\.position_embedding\.weight$": (f"{fp}encoder.position_embeddings", _Transform.DEFAULT),
        rf"{hp}embeddings\.patch_embedding\.weight$": (f"{fp}encoder.patch_embeddings.kernel", _Transform.CONV2D),
        rf"{hp}pre_layrnorm\.weight$": (f"{fp}encoder.ln_pre.scale", _Transform.DEFAULT),
        rf"{hp}pre_layrnorm\.bias$": (f"{fp}encoder.ln_pre.bias", _Transform.BIAS),
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
    }


def _text_mapping(flax_prefix: str, hf_prefix: str = "text_model") -> dict[str, tuple[str, _Transform]]:
    """Build regex key mapping from HuggingFace to Flax format for a CLIP text encoder.

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


def _infer_config(params_fstate: dict[str, Any]) -> dict[str, Any]:
    """Infer CLIP model configuration from weight shapes.

    Args:
        params_fstate (dict[str, Any]): Loaded parameter state dictionary.

    Returns:
        dict[str, Any]: Inferred configuration dictionary with text_config and vision_config.
    """
    text_hidden_size = params_fstate["text_model.embeddings.token_embedding.weight"].shape[1]
    text_max_pos = params_fstate["text_model.embeddings.position_embedding.weight"].shape[0]
    text_vocab_size = params_fstate["text_model.embeddings.token_embedding.weight"].shape[0]
    vision_hidden_size = params_fstate["vision_model.embeddings.class_embedding"].shape[0]
    vision_patch_size = params_fstate["vision_model.embeddings.patch_embedding.weight"].shape[2]
    pos_embed_len = params_fstate["vision_model.embeddings.position_embedding.weight"].shape[0]
    vision_image_size = int((pos_embed_len - 1) ** 0.5) * vision_patch_size
    text_num_layers = max(int(k.split(".")[3]) + 1 for k in params_fstate if k.startswith("text_model.encoder.layers.") and k.endswith(".self_attn.q_proj.weight"))
    vision_num_layers = max(int(k.split(".")[3]) + 1 for k in params_fstate if k.startswith("vision_model.encoder.layers.") and k.endswith(".self_attn.q_proj.weight"))
    return {
        "text_config": {
            "hidden_size": text_hidden_size,
            "num_attention_heads": text_hidden_size // 64,
            "num_hidden_layers": text_num_layers,
            "max_position_embeddings": text_max_pos,
            "vocab_size": text_vocab_size,
        },
        "vision_config": {
            "hidden_size": vision_hidden_size,
            "num_attention_heads": vision_hidden_size // 64,
            "num_hidden_layers": vision_num_layers,
            "image_size": vision_image_size,
            "patch_size": vision_patch_size,
        },
    }


def _create_config(model: "CLIP") -> dict[str, Any]:
    """Create HuggingFace config dictionary from CLIP model.

    Args:
        model (CLIP): CLIP model instance.

    Returns:
        dict[str, Any]: Configuration in HuggingFace format.
    """
    n_patches_plus_one = model.vision_model.encoder.position_embeddings[...].shape[1]
    img_size = int((n_patches_plus_one - 1) ** 0.5) * model.vision_patch_size
    return {
        "model_type": "clip",
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


def save_pretrained(model: "CLIP", save_directory: str) -> None:
    """Save CLIP model in HuggingFace format.

    Args:
        model (CLIP): CLIP model to save.
        save_directory (str): Output directory.
    """
    _SPECIAL_MAPPINGS = {
        "text_model.ln_final.weight": "text_model.final_layer_norm.weight",
        "text_model.ln_final.bias": "text_model.final_layer_norm.bias",
        "text_model.positional_embedding": "text_model.embeddings.position_embedding.weight",
        "text_model.token_embedding.embedding": "text_model.embeddings.token_embedding.weight",
        "text_model.text_projection.weight": "text_projection.weight",
        "vision_model.encoder.ln_pre.weight": "vision_model.pre_layrnorm.weight",
        "vision_model.encoder.ln_pre.bias": "vision_model.pre_layrnorm.bias",
        "vision_model.encoder.ln_post.weight": "vision_model.post_layernorm.weight",
        "vision_model.encoder.ln_post.bias": "vision_model.post_layernorm.bias",
        "vision_model.encoder.cls_token": "vision_model.embeddings.class_embedding",
        "vision_model.encoder.position_embeddings": "vision_model.embeddings.position_embedding.weight",
        "vision_model.encoder.patch_embeddings.weight": "vision_model.embeddings.patch_embedding.weight",
        "vision_model.vision_position_ids": "vision_model.embeddings.position_ids",
        "vision_model.visual_projection.weight": "visual_projection.weight",
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
    if jax.process_index() == 0:
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    _, state = nnx.split(model)
    state_dict = nnx.to_pure_dict(state)
    state_dict = expand_scanned_layers(state_dict)
    hf_state = convert_state_to_hf_format(state_dict, _SPECIAL_MAPPINGS, _SPECIAL_RENAMINGS)
    save_safetensors(hf_state, os.path.join(save_directory, "model.safetensors"))


def save_vision_pretrained(model: "CLIPVisionModel", save_directory: str) -> None:
    """Save CLIP vision model in HuggingFace format.

    Args:
        model (CLIPVisionModel): Model to save.
        save_directory (str): Output directory.
    """
    _SPECIAL_MAPPINGS = {
        "vision_model.encoder.ln_pre.weight": "vision_model.pre_layrnorm.weight",
        "vision_model.encoder.ln_pre.bias": "vision_model.pre_layrnorm.bias",
        "vision_model.encoder.ln_post.weight": "vision_model.post_layernorm.weight",
        "vision_model.encoder.ln_post.bias": "vision_model.post_layernorm.bias",
        "vision_model.encoder.cls_token": "vision_model.embeddings.class_embedding",
        "vision_model.encoder.position_embeddings": "vision_model.embeddings.position_embedding.weight",
        "vision_model.encoder.patch_embeddings.weight": "vision_model.embeddings.patch_embedding.weight",
        "vision_model.visual_projection.weight": "visual_projection.weight",
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

    n_patches_plus_one = model.encoder.position_embeddings[...].shape[1]
    img_size = int((n_patches_plus_one - 1) ** 0.5) * model.encoder.patch_embeddings.kernel_size[0]
    vision_config = {
        "hidden_size": model.vision_hidden_size,
        "image_size": img_size,
        "intermediate_size": model.vision_hidden_size * 4,
        "num_attention_heads": model.vision_hidden_size // 64,
        "num_hidden_layers": model.vision_layers,
        "patch_size": model.vision_patch_size,
        "projection_dim": model.projection_dim,
    }
    text_config = {"hidden_size": model.projection_dim}
    config = {"vision_config": vision_config, "text_config": text_config, "model_type": "clip"}

    if jax.process_index() == 0:
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    _, state = nnx.split(model)
    state_dict = expand_scanned_layers(nnx.to_pure_dict(state))

    prefixed_renamings: dict[str, str] = {}
    for k, v in _SPECIAL_RENAMINGS.items():
        if k.startswith("."):
            prefixed_renamings[k] = v
        else:
            prefixed_renamings["vision_model." + k] = "vision_model." + v

    hf_state = convert_state_to_hf_format({"vision_model": state_dict}, _SPECIAL_MAPPINGS, prefixed_renamings)
    save_safetensors(hf_state, os.path.join(save_directory, "model.safetensors"))


def save_text_pretrained(model: "CLIPTextModel", save_directory: str) -> None:
    """Save CLIP text model in HuggingFace format.

    Args:
        model (CLIPTextModel): Model to save.
        save_directory (str): Output directory.
    """
    _SPECIAL_MAPPINGS = {
        "text_model.ln_final.weight": "text_model.final_layer_norm.weight",
        "text_model.ln_final.bias": "text_model.final_layer_norm.bias",
        "text_model.positional_embedding": "text_model.embeddings.position_embedding.weight",
        "text_model.token_embedding.embedding": "text_model.embeddings.token_embedding.weight",
        "text_model.text_projection.weight": "text_projection.weight",
        "text_model.text_projection.bias": "text_projection.bias",
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
    config = {"text_config": text_config, "model_type": "clip"}

    if jax.process_index() == 0:
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

    hf_state = convert_state_to_hf_format({"text_model": state_dict}, _SPECIAL_MAPPINGS, prefixed_renamings)
    save_safetensors(hf_state, os.path.join(save_directory, "model.safetensors"))


def _apply_mapping(
    model: _M,
    params_fstate: dict[str, Any],
    mapping: dict[str, tuple[str, _Transform]],
    param_dtype: DTypeLike,
) -> _M:
    """Apply loaded HF parameters to model using regex mapping.

    Args:
        model (nnx.Module): Target model (created with real weights).
        params_fstate (dict[str, Any]): Loaded HuggingFace parameter dict.
        mapping (dict[str, tuple[str, _Transform]]): Regex key mapping.
        param_dtype (DTypeLike): Parameter dtype to cast tensors to.

    Returns:
        nnx.Module: Model with loaded parameters (same object, updated in-place).
    """
    flat_state = dict(nnx.to_flat_state(nnx.state(model, nnx.Param)))
    layer_accum: dict[tuple, dict[int, Any]] = {}
    for hf_key, tensor in params_fstate.items():
        jax_key, transform = map_to_bonsai_key(mapping, hf_key)
        if jax_key is None:
            continue
        keys = tuple(stoi(k) for k in jax_key.split("."))
        permute_rule, _, _ = transform.value
        t = tensor.astype(param_dtype)
        if permute_rule is not None:
            t = jnp.transpose(t, permute_rule)
        if keys in flat_state:
            var = flat_state[keys]
            if t.shape != var[...].shape:
                t = t.reshape(var[...].shape)
            var[...] = t
        else:
            batched_keys, layer_idx = to_scan_batched_keys(keys)
            if batched_keys is not None and batched_keys in flat_state:
                layer_accum.setdefault(batched_keys, {})[layer_idx] = t
    for batched_keys, layers_dict in layer_accum.items():
        var = flat_state[batched_keys]
        num_layers = var[...].shape[0]
        stacked = jnp.stack([layers_dict[i] for i in range(num_layers)], axis=0)
        if stacked.shape != var[...].shape:
            stacked = stacked.reshape(var[...].shape)
        var[...] = stacked
    nnx.update(model, nnx.from_flat_state(flat_state))
    return model


def load_text_from_pretrained(
    cls,
    model_name_or_path: str,
    use_pytorch: bool,
    rngs: rnglib.Rngs | None,
    dtype: DTypeLike,
    param_dtype: DTypeLike,
    sharding: ShardingSpec,
    use_gradient_checkpointing: bool,
) -> "CLIPTextModel":
    """Load pretrained CLIP text model.

    Args:
        cls: CLIPTextModel class.
        model_name_or_path (str): Model path or ID.
        use_pytorch (bool): Load from PyTorch.
        rngs (rnglib.Rngs | None): RNG state. If None, initializes to nnx.Rngs(0).
        dtype (DTypeLike): Computation dtype.
        param_dtype (DTypeLike): Parameter dtype.
        sharding (ShardingSpec): Sharding specification for parameters.
        use_gradient_checkpointing (bool): Enable gradient checkpointing.

    Returns:
        CLIPTextModel: Loaded model.
    """
    if rngs is None:
        rngs = nnx.Rngs(0)
    params_fstate, config_dict = load_params_and_config(model_name_or_path, use_pytorch)
    if not config_dict:
        config_dict = _infer_config(params_fstate)
    text_config = config_dict["text_config"]

    model = cls(
        context_length=text_config["max_position_embeddings"],
        vocab_size=text_config["vocab_size"],
        text_hidden_size=text_config["hidden_size"],
        num_text_heads=text_config["num_attention_heads"],
        num_text_layers=text_config["num_hidden_layers"],
        use_gradient_checkpointing=use_gradient_checkpointing,
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
        sharding=sharding,
    )

    mapping = {
        **_text_mapping(""),
        r"text_projection\.weight$": ("text_projection.kernel", _Transform.LINEAR),
    }
    return _apply_mapping(model, params_fstate, mapping, param_dtype)


def load_vision_from_pretrained(
    cls,
    model_name_or_path: str,
    use_pytorch: bool,
    rngs: rnglib.Rngs | None,
    dtype: DTypeLike,
    param_dtype: DTypeLike,
    sharding: ShardingSpec,
    use_gradient_checkpointing: bool,
) -> "CLIPVisionModel":
    """Load pretrained CLIP vision model.

    Args:
        cls: CLIPVisionModel class.
        model_name_or_path (str): Model path or ID.
        use_pytorch (bool): Load from PyTorch.
        rngs (rnglib.Rngs | None): RNG state. If None, initializes to nnx.Rngs(0).
        dtype (DTypeLike): Computation dtype.
        param_dtype (DTypeLike): Parameter dtype.
        sharding (ShardingSpec): Sharding specification for parameters.
        use_gradient_checkpointing (bool): Enable gradient checkpointing.

    Returns:
        CLIPVisionModel: Loaded model.
    """
    if rngs is None:
        rngs = nnx.Rngs(0)
    params_fstate, config_dict = load_params_and_config(model_name_or_path, use_pytorch)
    if not config_dict:
        config_dict = _infer_config(params_fstate)
    vision_config = config_dict["vision_config"]
    text_config = config_dict["text_config"]

    model = cls(
        image_resolution=vision_config["image_size"],
        vision_layers=vision_config["num_hidden_layers"],
        vision_hidden_size=vision_config["hidden_size"],
        vision_patch_size=vision_config["patch_size"],
        projection_dim=text_config["hidden_size"],
        use_gradient_checkpointing=use_gradient_checkpointing,
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
        sharding=sharding,
    )

    mapping = {
        **_vision_mapping(""),
        r"visual_projection\.weight$": ("visual_projection.kernel", _Transform.LINEAR),
    }
    return _apply_mapping(model, params_fstate, mapping, param_dtype)


def load_from_pretrained(
    cls,
    model_name_or_path: str,
    use_pytorch: bool,
    rngs: rnglib.Rngs | None,
    dtype: DTypeLike,
    param_dtype: DTypeLike,
    sharding: ShardingSpec,
    use_gradient_checkpointing: bool,
) -> "CLIP":
    """Load pretrained CLIP model.

    Args:
        cls: CLIP class.
        model_name_or_path (str): Model path or ID.
        use_pytorch (bool): Load from PyTorch.
        rngs (rnglib.Rngs | None): RNG state. If None, initializes to nnx.Rngs(0).
        dtype (DTypeLike): Computation dtype.
        param_dtype (DTypeLike): Parameter dtype.
        sharding (ShardingSpec): Sharding specification for parameters.
        use_gradient_checkpointing (bool): Enable gradient checkpointing.

    Returns:
        CLIP: Loaded model.
    """
    if rngs is None:
        rngs = nnx.Rngs(0)
    params_fstate, config_dict = load_params_and_config(model_name_or_path, use_pytorch)
    if not config_dict:
        config_dict = _infer_config(params_fstate)
    text_config = config_dict["text_config"]
    vision_config = config_dict["vision_config"]

    model = cls(
        image_resolution=vision_config["image_size"],
        vision_layers=vision_config["num_hidden_layers"],
        vision_hidden_size=vision_config["hidden_size"],
        vision_patch_size=vision_config["patch_size"],
        context_length=text_config["max_position_embeddings"],
        vocab_size=text_config["vocab_size"],
        text_hidden_size=text_config["hidden_size"],
        num_text_heads=text_config["num_attention_heads"],
        num_text_layers=text_config["num_hidden_layers"],
        use_gradient_checkpointing=use_gradient_checkpointing,
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
        sharding=sharding,
    )

    mapping = {
        r"logit_scale$": ("logit_scale", _Transform.DEFAULT),
        r"visual_projection\.weight$": ("vision_model.visual_projection.kernel", _Transform.LINEAR),
        r"text_projection\.weight$": ("text_model.text_projection.kernel", _Transform.LINEAR),
        **_vision_mapping("vision_model"),
        **_text_mapping("text_model"),
    }
    m = _apply_mapping(model, params_fstate, mapping, param_dtype)
    m._original_config = config_dict
    return m
