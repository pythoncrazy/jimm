import json
import os
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib
from jax.sharding import Mesh
from jaxtyping import DTypeLike
from safetensors.flax import save_file as save_safetensors

from jimm.common.loading_utils import (
    build_base_text_mapping,
    build_base_vision_mapping,
    load_and_apply_params,
    load_params_and_config,
)
from jimm.common.utils import convert_state_to_hf_format

if TYPE_CHECKING:
    from jimm.models import SigLIP, SigLIPTextModel, SigLIPVisionModel


def _transform_param(
    value: Any,
    flax_key: tuple,
    hf_key: tuple,
    config: dict[str, Any],
) -> Any:
    """Transform parameter from HuggingFace to Flax format for SigLIP.

    Args:
        value (Any): Parameter value from HuggingFace.
        flax_key (tuple): Flax parameter key.
        hf_key (tuple): HuggingFace parameter key.
        config (dict[str, Any]): Model configuration.

    Returns:
        Any: Transformed parameter in Flax format.
    """
    hidden_size = config.get("hidden_size", 768)
    num_heads = config.get("num_attention_heads", hidden_size // 64)
    head_dim = hidden_size // num_heads

    if "patch_embeddings" in flax_key and flax_key[-1] == "kernel":
        return jnp.transpose(value, (2, 3, 1, 0))
    if flax_key[-1] == "position_embeddings":
        return value.reshape(1, value.shape[0], value.shape[1])
    if flax_key[-1] == "positional_embedding":
        return value
    if flax_key[-1] == "probe":
        return value.reshape(1, 1, hidden_size)
    if flax_key[-1] in ("logit_scale", "logit_bias"):
        return jnp.squeeze(value)
    if hf_key[-1] == "in_proj_weight":
        q_w, k_w, v_w = jnp.split(value, 3, axis=0)
        w_map = {"query": q_w, "key": k_w, "value": v_w}
        return jnp.transpose(w_map[flax_key[-2]], (1, 0)).reshape(hidden_size, num_heads, head_dim)
    if hf_key[-1] == "in_proj_bias":
        q_b, k_b, v_b = jnp.split(value, 3, axis=0)
        b_map = {"query": q_b, "key": k_b, "value": v_b}
        return b_map[flax_key[-2]].reshape(num_heads, head_dim)
    if len(hf_key) >= 2 and hf_key[-2:] in (("q_proj", "weight"), ("k_proj", "weight"), ("v_proj", "weight")):
        return jnp.transpose(value, (1, 0)).reshape(hidden_size, num_heads, head_dim)
    if len(hf_key) >= 2 and hf_key[-2] in ("q_proj", "k_proj", "v_proj") and hf_key[-1] == "bias":
        return value.reshape(num_heads, head_dim)
    if len(hf_key) >= 2 and hf_key[-2:] == ("out_proj", "weight"):
        return jnp.transpose(value, (1, 0)).reshape(num_heads, head_dim, hidden_size)
    if len(hf_key) >= 2 and hf_key[-2:] == ("token_embedding", "weight"):
        return value
    if "position_embedding" in hf_key and hf_key[-1] == "weight":
        return value
    if hf_key[-1] == "weight" and value.ndim == 2:
        return jnp.transpose(value, (1, 0))
    return value


def _build_param_mapping(
    config: dict[str, Any],
    component: str = "both",
    text_config: dict[str, Any] | None = None,
    vision_config: dict[str, Any] | None = None,
) -> dict[tuple, tuple]:
    """Build parameter mapping for loading SigLIP models.

    Args:
        config (dict[str, Any]): Model configuration.
        component (str): Component to map ('text', 'vision', or 'both').
        text_config (dict[str, Any] | None): Text config for full model.
        vision_config (dict[str, Any] | None): Vision config for full model.

    Returns:
        dict[tuple, tuple]: Parameter mapping.
    """
    mapping = {}

    if component == "text":
        mapping.update(build_base_text_mapping(config))
        mapping[("text_projection", "kernel")] = ("text_model", "head", "weight")
        mapping[("text_projection", "bias")] = ("text_model", "head", "bias")

    elif component == "vision":
        mapping.update(build_base_vision_mapping(config, prefix=""))
        mapping[("encoder", "patch_embeddings", "bias")] = ("vision_model", "embeddings", "patch_embedding", "bias")
        probe_keys = [
            (("encoder", "MAPHead", "probe"), ("vision_model", "head", "probe")),
            (("encoder", "MAPHead", "layernorm", "scale"), ("vision_model", "head", "layernorm", "weight")),
            (("encoder", "MAPHead", "layernorm", "bias"), ("vision_model", "head", "layernorm", "bias")),
            (("encoder", "MAPHead", "mlp", "layers", 0, "kernel"), ("vision_model", "head", "mlp", "fc1", "weight")),
            (("encoder", "MAPHead", "mlp", "layers", 0, "bias"), ("vision_model", "head", "mlp", "fc1", "bias")),
            (("encoder", "MAPHead", "mlp", "layers", 2, "kernel"), ("vision_model", "head", "mlp", "fc2", "weight")),
            (("encoder", "MAPHead", "mlp", "layers", 2, "bias"), ("vision_model", "head", "mlp", "fc2", "bias")),
            (("encoder", "MAPHead", "attn", "query", "kernel"), ("vision_model", "head", "attention", "in_proj_weight")),
            (("encoder", "MAPHead", "attn", "query", "bias"), ("vision_model", "head", "attention", "in_proj_bias")),
            (("encoder", "MAPHead", "attn", "key", "kernel"), ("vision_model", "head", "attention", "in_proj_weight")),
            (("encoder", "MAPHead", "attn", "key", "bias"), ("vision_model", "head", "attention", "in_proj_bias")),
            (("encoder", "MAPHead", "attn", "value", "kernel"), ("vision_model", "head", "attention", "in_proj_weight")),
            (("encoder", "MAPHead", "attn", "value", "bias"), ("vision_model", "head", "attention", "in_proj_bias")),
            (("encoder", "MAPHead", "attn", "out", "kernel"), ("vision_model", "head", "attention", "out_proj", "weight")),
            (("encoder", "MAPHead", "attn", "out", "bias"), ("vision_model", "head", "attention", "out_proj", "bias")),
        ]
        for flax_key, hf_key in probe_keys:
            mapping[flax_key] = hf_key

    elif component == "both":
        mapping[("logit_scale",)] = ("logit_scale",)
        mapping[("logit_bias",)] = ("logit_bias",)

        text_mapping = build_base_text_mapping(text_config, prefix="text_model")
        for k, v in text_mapping.items():
            mapping[k] = v

        mapping[("text_model", "text_projection", "kernel")] = ("text_model", "head", "weight")
        mapping[("text_model", "text_projection", "bias")] = ("text_model", "head", "bias")

        vision_mapping = build_base_vision_mapping(vision_config, prefix="vision_model")
        for k, v in vision_mapping.items():
            mapping[k] = v

        mapping[("vision_model", "encoder", "patch_embeddings", "bias")] = ("vision_model", "embeddings", "patch_embedding", "bias")
        probe_keys = [
            (("vision_model", "encoder", "MAPHead", "probe"), ("vision_model", "head", "probe")),
            (("vision_model", "encoder", "MAPHead", "layernorm", "scale"), ("vision_model", "head", "layernorm", "weight")),
            (("vision_model", "encoder", "MAPHead", "layernorm", "bias"), ("vision_model", "head", "layernorm", "bias")),
            (("vision_model", "encoder", "MAPHead", "mlp", "layers", 0, "kernel"), ("vision_model", "head", "mlp", "fc1", "weight")),
            (("vision_model", "encoder", "MAPHead", "mlp", "layers", 0, "bias"), ("vision_model", "head", "mlp", "fc1", "bias")),
            (("vision_model", "encoder", "MAPHead", "mlp", "layers", 2, "kernel"), ("vision_model", "head", "mlp", "fc2", "weight")),
            (("vision_model", "encoder", "MAPHead", "mlp", "layers", 2, "bias"), ("vision_model", "head", "mlp", "fc2", "bias")),
            (("vision_model", "encoder", "MAPHead", "attn", "query", "kernel"), ("vision_model", "head", "attention", "in_proj_weight")),
            (("vision_model", "encoder", "MAPHead", "attn", "query", "bias"), ("vision_model", "head", "attention", "in_proj_bias")),
            (("vision_model", "encoder", "MAPHead", "attn", "key", "kernel"), ("vision_model", "head", "attention", "in_proj_weight")),
            (("vision_model", "encoder", "MAPHead", "attn", "key", "bias"), ("vision_model", "head", "attention", "in_proj_bias")),
            (("vision_model", "encoder", "MAPHead", "attn", "value", "kernel"), ("vision_model", "head", "attention", "in_proj_weight")),
            (("vision_model", "encoder", "MAPHead", "attn", "value", "bias"), ("vision_model", "head", "attention", "in_proj_bias")),
            (("vision_model", "encoder", "MAPHead", "attn", "out", "kernel"), ("vision_model", "head", "attention", "out_proj", "weight")),
            (("vision_model", "encoder", "MAPHead", "attn", "out", "bias"), ("vision_model", "head", "attention", "out_proj", "bias")),
        ]
        for flax_key, hf_key in probe_keys:
            mapping[flax_key] = hf_key

    return mapping


def _create_transform_fn(config: dict[str, Any]):
    """Create parameter transformation function for SigLIP.

    Args:
        config (dict[str, Any]): Model configuration.

    Returns:
        Callable: Transformation function.
    """

    def transform_fn(value: Any, flax_key: tuple, hf_key: tuple) -> Any:
        text_config = config.get("text_config", config)
        vision_config = config.get("vision_config", config)

        if "vision_model" in flax_key or "vision_model" in hf_key:
            return _transform_param(value, flax_key, hf_key, vision_config)
        else:
            return _transform_param(value, flax_key, hf_key, text_config)

    return transform_fn


def _create_config(model: "SigLIP") -> dict[str, Any]:
    """Create HuggingFace config dictionary.

    Args:
        model (SigLIP): SigLIP model instance.

    Returns:
        dict[str, Any]: Configuration in HuggingFace format.
    """
    return {
        "model_type": "siglip",
        "text_config": {
            "hidden_size": model.transformer_width,
            "num_attention_heads": model.transformer_heads,
            "num_hidden_layers": model.transformer_layers,
            "max_position_embeddings": model.context_length,
            "vocab_size": model.vocab_size,
        },
        "vision_config": {
            "hidden_size": model.vision_width,
            "num_attention_heads": model.vision_width // 64,
            "num_hidden_layers": model.vision_layers,
            "image_size": model.vision_model.encoder.img_size,
            "patch_size": model.vision_patch_size,
        },
    }


def save_pretrained(model: "SigLIP", save_directory: str):
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
        "vision_model.encoder.MAPHead.probe": "vision_model.head.probe",
        "vision_model.encoder.MAPHead.layernorm.weight": "vision_model.head.layernorm.weight",
        "vision_model.encoder.MAPHead.layernorm.bias": "vision_model.head.layernorm.bias",
        "vision_model.encoder.MAPHead.mlp.fc1.weight": "vision_model.head.mlp.fc1.weight",
        "vision_model.encoder.MAPHead.mlp.fc1.bias": "vision_model.head.mlp.fc1.bias",
        "vision_model.encoder.MAPHead.mlp.layers.2.weight": "vision_model.head.mlp.fc2.weight",
        "vision_model.encoder.MAPHead.mlp.layers.2.bias": "vision_model.head.mlp.fc2.bias",
        "vision_model.encoder.MAPHead.attn.in_proj_weight": "vision_model.head.attention.in_proj_weight",
        "vision_model.encoder.MAPHead.attn.in_proj_bias": "vision_model.head.attention.in_proj_bias",
        "vision_model.encoder.MAPHead.self_attn.out_proj.weight": "vision_model.head.attention.out_proj.weight",
        "vision_model.encoder.MAPHead.self_attn.out_proj.bias": "vision_model.head.attention.out_proj.bias",
    }
    _SPECIAL_RENAMINGS = {
        "text_model.transformer.layers": "text_model.encoder.layers",
        "vision_model.encoder.encoder.layers": "vision_model.encoder.layers",
        ".attn.query.": ".self_attn.q_proj.",
        ".attn.key.": ".self_attn.k_proj.",
        ".attn.value.": ".self_attn.v_proj.",
        ".attn.out.": ".self_attn.out_proj.",
        ".mlp.layers.0.": ".mlp.fc1.",
        ".mlp.layers.3.": ".mlp.fc2.",
        ".norm1.": ".layer_norm1.",
        ".norm2.": ".layer_norm2.",
    }
    os.makedirs(save_directory, exist_ok=True)

    config = model._original_config.copy() if model._original_config else _create_config(model)
    with open(os.path.join(save_directory, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    _, state = nnx.split(model)
    state_dict = nnx.to_pure_dict(state)

    if "vision_model" in state_dict and "MAPHead" in state_dict["vision_model"]["encoder"]:
        maphead = state_dict["vision_model"]["encoder"]["MAPHead"]
        if "attn" in maphead:
            attn = maphead["attn"]

            if all(k in attn for k in ["query", "key", "value"]):
                q_weight = attn["query"]["kernel"]
                k_weight = attn["key"]["kernel"]
                v_weight = attn["value"]["kernel"]

                q_flat = q_weight.reshape(q_weight.shape[0], -1).T
                k_flat = k_weight.reshape(k_weight.shape[0], -1).T
                v_flat = v_weight.reshape(v_weight.shape[0], -1).T

                in_proj_weight = jnp.concatenate([q_flat, k_flat, v_flat], axis=0)

                q_bias = attn["query"]["bias"]
                k_bias = attn["key"]["bias"]
                v_bias = attn["value"]["bias"]

                q_bias_flat = q_bias.flatten()
                k_bias_flat = k_bias.flatten()
                v_bias_flat = v_bias.flatten()

                in_proj_bias = jnp.concatenate([q_bias_flat, k_bias_flat, v_bias_flat], axis=0)

                del attn["query"]
                del attn["key"]
                del attn["value"]

                attn["in_proj_weight"] = in_proj_weight
                attn["in_proj_bias"] = in_proj_bias

            if "out" in attn:
                out_weight = attn["out"]["kernel"]
                out_bias = attn["out"]["bias"]

                out_weight_flat = out_weight.reshape(-1, out_weight.shape[-1])

                del attn["out"]
                if "self_attn" not in maphead:
                    maphead["self_attn"] = {}
                if "out_proj" not in maphead["self_attn"]:
                    maphead["self_attn"]["out_proj"] = {}
                maphead["self_attn"]["out_proj"]["weight"] = out_weight_flat
                maphead["self_attn"]["out_proj"]["bias"] = out_bias

    hf_state = convert_state_to_hf_format(state_dict, _SPECIAL_MAPPINGS, _SPECIAL_RENAMINGS)

    for key in ["logit_scale", "logit_bias"]:
        if key in hf_state and hf_state[key].ndim == 0:
            hf_state[key] = jnp.expand_dims(hf_state[key], 0)

    hf_state.pop("vision_model.encoder.vision_position_ids", None)

    save_safetensors(hf_state, os.path.join(save_directory, "model.safetensors"))


def load_text_from_pretrained(
    cls,
    model_name_or_path: str,
    use_pytorch: bool,
    mesh: Mesh | None,
    dtype: DTypeLike,
    param_dtype: DTypeLike,
    use_gradient_checkpointing: bool,
    rngs: rnglib.Rngs,
) -> "SigLIPTextModel":
    """Load pretrained SigLIP text model.

    Args:
        cls: SigLIPTextModel class.
        model_name_or_path (str): Model path or ID.
        use_pytorch (bool): Load from PyTorch.
        mesh (Mesh | None): Device mesh.
        dtype (DTypeLike): Computation dtype.
        param_dtype (DTypeLike): Parameter dtype.
        use_gradient_checkpointing (bool): Enable gradient checkpointing.
        rngs (rnglib.Rngs): RNG state.

    Returns:
        SigLIPTextModel: Loaded model.
    """
    params_fstate, config_dict = load_params_and_config(model_name_or_path, use_pytorch)

    context_length = params_fstate["text_model.embeddings.position_embedding.weight"].shape[0]
    vocab_size = params_fstate["text_model.embeddings.token_embedding.weight"].shape[0]
    text_hidden_size = params_fstate["text_model.embeddings.token_embedding.weight"].shape[1]
    text_num_layers = 0
    for k_param in params_fstate:
        if k_param.startswith("text_model.encoder.layers.") and k_param.endswith(".self_attn.q_proj.weight"):
            layer_idx = int(k_param.split(".")[3])
            text_num_layers = max(text_num_layers, layer_idx + 1)

    text_config = {
        "hidden_size": text_hidden_size,
        "num_attention_heads": text_hidden_size // 64,
        "num_hidden_layers": text_num_layers,
        "max_position_embeddings": context_length,
        "vocab_size": vocab_size,
    }

    model = cls(
        context_length=context_length,
        vocab_size=vocab_size,
        transformer_width=text_hidden_size,
        transformer_heads=text_hidden_size // 64,
        transformer_layers=text_num_layers,
        use_gradient_checkpointing=use_gradient_checkpointing,
        mesh=mesh,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
    )

    mapping = _build_param_mapping(text_config, component="text")
    transform_fn = _create_transform_fn(text_config)
    load_and_apply_params(model, params_fstate, mapping, transform_fn, param_dtype)

    return model


def load_vision_from_pretrained(
    cls,
    model_name_or_path: str,
    use_pytorch: bool,
    mesh: Mesh | None,
    dtype: DTypeLike,
    param_dtype: DTypeLike,
    use_gradient_checkpointing: bool,
    rngs: rnglib.Rngs,
) -> "SigLIPVisionModel":
    """Load pretrained SigLIP vision model.

    Args:
        cls: SigLIPVisionModel class.
        model_name_or_path (str): Model path or ID.
        use_pytorch (bool): Load from PyTorch.
        mesh (Mesh | None): Device mesh.
        dtype (DTypeLike): Computation dtype.
        param_dtype (DTypeLike): Parameter dtype.
        use_gradient_checkpointing (bool): Enable gradient checkpointing.
        rngs (rnglib.Rngs): RNG state.

    Returns:
        SigLIPVisionModel: Loaded model.
    """
    params_fstate, config_dict = load_params_and_config(model_name_or_path, use_pytorch)

    vision_patch_size = params_fstate["vision_model.embeddings.patch_embedding.weight"].shape[3]
    vision_width = params_fstate["vision_model.embeddings.patch_embedding.bias"].shape[0]
    vision_num_layers = 0
    for k in params_fstate:
        if k.startswith("vision_model.encoder.layers.") and k.endswith(".mlp.fc2.bias"):
            vision_num_layers = max(vision_num_layers, int(k.split(".")[3]) + 1)

    vision_config = {
        "hidden_size": vision_width,
        "num_attention_heads": vision_width // 64,
        "num_hidden_layers": vision_num_layers,
        "image_size": config_dict["vision_config"]["image_size"],
        "patch_size": vision_patch_size,
    }

    model = cls(
        image_resolution=config_dict["vision_config"]["image_size"],
        vision_layers=vision_num_layers,
        vision_width=vision_width,
        vision_patch_size=vision_patch_size,
        use_gradient_checkpointing=use_gradient_checkpointing,
        mesh=mesh,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
    )

    mapping = _build_param_mapping(vision_config, component="vision")
    transform_fn = _create_transform_fn(vision_config)
    load_and_apply_params(model, params_fstate, mapping, transform_fn, param_dtype)

    return model


def load_from_pretrained(
    cls,
    model_name_or_path: str,
    use_pytorch: bool,
    mesh: Mesh | None,
    dtype: DTypeLike,
    param_dtype: DTypeLike,
    use_gradient_checkpointing: bool,
    rngs: rnglib.Rngs,
) -> "SigLIP":
    """Load pretrained SigLIP model.

    Args:
        cls: SigLIP class.
        model_name_or_path (str): Model path or ID.
        use_pytorch (bool): Load from PyTorch.
        mesh (Mesh | None): Device mesh.
        dtype (DTypeLike): Computation dtype.
        param_dtype (DTypeLike): Parameter dtype.
        use_gradient_checkpointing (bool): Enable gradient checkpointing.
        rngs (rnglib.Rngs): RNG state.

    Returns:
        SigLIP: Loaded model.
    """
    params_fstate, config_dict = load_params_and_config(model_name_or_path, use_pytorch)

    vision_patch_size = params_fstate["vision_model.embeddings.patch_embedding.weight"].shape[3]
    vision_width = params_fstate["vision_model.embeddings.patch_embedding.bias"].shape[0]
    vision_num_layers = 0
    for k in params_fstate:
        if k.startswith("vision_model.encoder.layers.") and k.endswith(".mlp.fc2.bias"):
            vision_num_layers = max(vision_num_layers, int(k.split(".")[3]) + 1)

    context_length = params_fstate["text_model.embeddings.position_embedding.weight"].shape[0]
    vocab_size = params_fstate["text_model.embeddings.token_embedding.weight"].shape[0]
    text_hidden_size = params_fstate["text_model.embeddings.token_embedding.weight"].shape[1]
    text_num_layers = 0
    for k_param in params_fstate:
        if k_param.startswith("text_model.encoder.layers.") and k_param.endswith(".self_attn.q_proj.weight"):
            layer_idx = int(k_param.split(".")[3])
            text_num_layers = max(text_num_layers, layer_idx + 1)

    text_config = {
        "hidden_size": text_hidden_size,
        "num_attention_heads": text_hidden_size // 64,
        "num_hidden_layers": text_num_layers,
        "max_position_embeddings": context_length,
        "vocab_size": vocab_size,
    }

    vision_config = {
        "hidden_size": vision_width,
        "num_attention_heads": vision_width // 64,
        "num_hidden_layers": vision_num_layers,
        "image_size": config_dict["vision_config"]["image_size"],
        "patch_size": vision_patch_size,
    }

    model = cls(
        image_resolution=config_dict["vision_config"]["image_size"],
        vision_layers=vision_num_layers,
        vision_width=vision_width,
        vision_patch_size=vision_patch_size,
        context_length=context_length,
        vocab_size=vocab_size,
        transformer_width=text_hidden_size,
        transformer_heads=text_hidden_size // 64,
        transformer_layers=text_num_layers,
        use_gradient_checkpointing=use_gradient_checkpointing,
        mesh=mesh,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
    )

    mapping = _build_param_mapping(
        {"text_config": text_config, "vision_config": vision_config},
        component="both",
        text_config=text_config,
        vision_config=vision_config,
    )
    transform_fn = _create_transform_fn({"text_config": text_config, "vision_config": vision_config})
    load_and_apply_params(model, params_fstate, mapping, transform_fn, param_dtype)

    model._original_config = config_dict
    return model
