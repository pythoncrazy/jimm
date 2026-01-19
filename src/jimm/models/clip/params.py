import json
import os
from typing import TYPE_CHECKING, Any

import jax
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
from jimm.common.splash_attention import SplashAttentionConfig
from jimm.common.utils import convert_state_to_hf_format

if TYPE_CHECKING:
    from jimm.models import CLIP, CLIPTextModel, CLIPVisionModel


def _transform_param(
    value: Any,
    flax_key: tuple,
    hf_key: tuple,
    config: dict[str, Any],
) -> Any:
    """Transform parameter from HuggingFace to Flax format.

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
    if flax_key[-1] == "cls_token":
        return value.reshape(1, 1, -1)
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
    text_width: int | None = None,
) -> dict[tuple, tuple]:
    """Build parameter mapping for loading CLIP models.

    Args:
        config (dict[str, Any]): Model configuration.
        component (str): Component to map ('text', 'vision', or 'both').
        text_config (dict[str, Any] | None): Text config for full model.
        vision_config (dict[str, Any] | None): Vision config for full model.
        text_width (int | None): Text width for vision-only models.

    Returns:
        dict[tuple, tuple]: Parameter mapping.
    """
    mapping = {}

    if component == "text":
        mapping.update(build_base_text_mapping(config))
        mapping[("text_projection", "kernel")] = ("text_projection", "weight")

    elif component == "vision":
        mapping.update(build_base_vision_mapping(config, prefix=""))
        mapping[("encoder", "cls_token")] = ("vision_model", "embeddings", "class_embedding")
        mapping[("encoder", "ln_pre", "scale")] = ("vision_model", "pre_layrnorm", "weight")
        mapping[("encoder", "ln_pre", "bias")] = ("vision_model", "pre_layrnorm", "bias")
        if text_width:
            mapping[("visual_projection", "kernel")] = ("visual_projection", "weight")

    elif component == "both":
        if text_config is None or vision_config is None:
            raise ValueError("text_config and vision_config must be provided when component='both'")

        mapping[("logit_scale",)] = ("logit_scale",)

        text_mapping = build_base_text_mapping(text_config, prefix="text_model")
        for k, v in text_mapping.items():
            mapping[k] = v

        mapping[("text_model", "text_projection", "kernel")] = ("text_projection", "weight")

        vision_mapping = build_base_vision_mapping(vision_config, prefix="vision_model")
        for k, v in vision_mapping.items():
            mapping[k] = v

        mapping[("vision_model", "encoder", "cls_token")] = ("vision_model", "embeddings", "class_embedding")
        mapping[("vision_model", "encoder", "ln_pre", "scale")] = ("vision_model", "pre_layrnorm", "weight")
        mapping[("vision_model", "encoder", "ln_pre", "bias")] = ("vision_model", "pre_layrnorm", "bias")
        mapping[("vision_model", "visual_projection", "kernel")] = ("visual_projection", "weight")

    return mapping


def _create_transform_fn(config: dict[str, Any]):
    """Create parameter transformation function for CLIP.

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


def _infer_config_from_params(params_fstate: dict[str, Any], use_pytorch: bool) -> dict[str, Any]:
    """Infer CLIP model configuration from weights.

    Args:
        params_fstate (dict[str, Any]): Parameter state dictionary.
        use_pytorch (bool): Whether loading from PyTorch format.

    Returns:
        dict[str, Any]: Inferred configuration dictionary.
    """
    if use_pytorch:
        raise ValueError("Configuration could not be loaded for PyTorch model")

    text_hidden_size = params_fstate["text_model.embeddings.token_embedding.weight"].shape[1]
    text_max_pos_embed = params_fstate["text_model.embeddings.position_embedding.weight"].shape[0]
    text_vocab_size = params_fstate["text_model.embeddings.token_embedding.weight"].shape[0]
    vision_hidden_size = params_fstate["vision_model.embeddings.class_embedding"].shape[0]
    vision_patch_size = params_fstate["vision_model.embeddings.patch_embedding.weight"].shape[2]
    pos_embed_shape = params_fstate["vision_model.embeddings.position_embedding.weight"].shape[0]
    vision_image_size = int((pos_embed_shape - 1) ** 0.5) * vision_patch_size

    text_num_layers = max(int(k.split(".")[3]) + 1 for k in params_fstate if k.startswith("text_model.encoder.layers.") and k.endswith(".self_attn.q_proj.weight"))
    vision_num_layers = max(int(k.split(".")[3]) + 1 for k in params_fstate if k.startswith("vision_model.encoder.layers.") and k.endswith(".self_attn.q_proj.weight"))

    return {
        "text_config": {
            "hidden_size": text_hidden_size,
            "num_attention_heads": text_hidden_size // 64,
            "num_hidden_layers": text_num_layers,
            "max_position_embeddings": text_max_pos_embed,
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
    """Create HuggingFace config dictionary.

    Args:
        model (CLIP): CLIP model instance.

    Returns:
        dict[str, Any]: Configuration in HuggingFace format.
    """
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
            "image_size": model.vision_model.encoder.img_size,
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
    # Fix layer numbering: layers_0 -> layers.0
    for i in range(100):
        _SPECIAL_RENAMINGS[f"layers_{i}."] = f"layers.{i}."
    os.makedirs(save_directory, exist_ok=True)

    config = model._original_config.copy() if model._original_config else _create_config(model)
    if jax.process_index() == 0:
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    _, state = nnx.split(model)
    hf_state = convert_state_to_hf_format(nnx.to_pure_dict(state), _SPECIAL_MAPPINGS, _SPECIAL_RENAMINGS)
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
    _SPECIAL_RENAMINGS = {
        "encoder.encoder.layers": "encoder.layers",
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

    vision_config = {
        "hidden_size": model.vision_hidden_size,
        "image_size": model.encoder.patch_embeddings.kernel_size[0] * int(model.encoder.position_embeddings[...].shape[1] ** 0.5),
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
    state_dict = nnx.to_pure_dict(state)

    prefixed_renamings = {}
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
    _SPECIAL_RENAMINGS = {
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
    state_dict = nnx.to_pure_dict(state)

    prefixed_renamings = {}
    for k, v in _SPECIAL_RENAMINGS.items():
        if k.startswith("."):
            prefixed_renamings[k] = v
        else:
            prefixed_renamings["text_model." + k] = "text_model." + v

    hf_state = convert_state_to_hf_format({"text_model": state_dict}, _SPECIAL_MAPPINGS, prefixed_renamings)
    save_safetensors(hf_state, os.path.join(save_directory, "model.safetensors"))


def load_text_from_pretrained(
    cls,
    model_name_or_path: str,
    use_pytorch: bool,
    rngs: rnglib.Rngs | None,
    dtype: DTypeLike,
    param_dtype: DTypeLike,
    mesh: Mesh | None,
    use_gradient_checkpointing: bool,
    splash_attention_config: SplashAttentionConfig | None = None,
) -> "CLIPTextModel":
    """Load pretrained CLIP text model.

    Args:
        cls: CLIPTextModel class.
        model_name_or_path (str): Model path or ID.
        use_pytorch (bool): Load from PyTorch.
        rngs (rnglib.Rngs | None): RNG state. If None, initializes to nnx.Rngs(0).
        dtype (DTypeLike): Computation dtype.
        param_dtype (DTypeLike): Parameter dtype.
        mesh (Mesh | None): Device mesh.
        use_gradient_checkpointing (bool): Enable gradient checkpointing.
        splash_attention_config (SplashAttentionConfig | None): Configuration for TPU splash attention. Defaults to None.

    Returns:
        CLIPTextModel: Loaded model.
    """
    if rngs is None:
        rngs = nnx.Rngs(0)
    params_fstate, config_dict = load_params_and_config(model_name_or_path, use_pytorch)

    if not config_dict:
        config_dict = _infer_config_from_params(params_fstate, use_pytorch)

    text_config = config_dict["text_config"]

    model = cls(
        context_length=text_config["max_position_embeddings"],
        vocab_size=text_config["vocab_size"],
        text_hidden_size=text_config["hidden_size"],
        num_text_heads=text_config["num_attention_heads"],
        num_text_layers=text_config["num_hidden_layers"],
        use_gradient_checkpointing=use_gradient_checkpointing,
        splash_attention_config=splash_attention_config,
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
        mesh=mesh,
    )

    mapping = _build_param_mapping(text_config, component="text")
    transform_fn = _create_transform_fn(text_config)
    load_and_apply_params(model, params_fstate, mapping, transform_fn, param_dtype)

    return model


def load_vision_from_pretrained(
    cls,
    model_name_or_path: str,
    use_pytorch: bool,
    rngs: rnglib.Rngs | None,
    dtype: DTypeLike,
    param_dtype: DTypeLike,
    mesh: Mesh | None,
    use_gradient_checkpointing: bool,
    splash_attention_config: SplashAttentionConfig | None = None,
) -> "CLIPVisionModel":
    """Load pretrained CLIP vision model.

    Args:
        cls: CLIPVisionModel class.
        model_name_or_path (str): Model path or ID.
        use_pytorch (bool): Load from PyTorch.
        rngs (rnglib.Rngs | None): RNG state. If None, initializes to nnx.Rngs(0).
        dtype (DTypeLike): Computation dtype.
        param_dtype (DTypeLike): Parameter dtype.
        mesh (Mesh | None): Device mesh.
        use_gradient_checkpointing (bool): Enable gradient checkpointing.
        splash_attention_config (SplashAttentionConfig | None): Configuration for TPU splash attention. Defaults to None.

    Returns:
        CLIPVisionModel: Loaded model.
    """
    if rngs is None:
        rngs = nnx.Rngs(0)
    params_fstate, config_dict = load_params_and_config(model_name_or_path, use_pytorch)

    if not config_dict:
        config_dict = _infer_config_from_params(params_fstate, use_pytorch)

    vision_config = config_dict["vision_config"]
    text_config = config_dict["text_config"]

    model = cls(
        image_resolution=vision_config["image_size"],
        vision_layers=vision_config["num_hidden_layers"],
        vision_hidden_size=vision_config["hidden_size"],
        vision_patch_size=vision_config["patch_size"],
        projection_dim=text_config["hidden_size"],
        use_gradient_checkpointing=use_gradient_checkpointing,
        splash_attention_config=splash_attention_config,
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
        mesh=mesh,
    )

    mapping = _build_param_mapping(vision_config, component="vision", text_width=text_config["hidden_size"])
    transform_fn = _create_transform_fn(vision_config)
    load_and_apply_params(model, params_fstate, mapping, transform_fn, param_dtype)

    return model


def load_from_pretrained(
    cls,
    model_name_or_path: str,
    use_pytorch: bool,
    rngs: rnglib.Rngs | None,
    dtype: DTypeLike,
    param_dtype: DTypeLike,
    mesh: Mesh | None,
    use_gradient_checkpointing: bool,
    splash_attention_config: SplashAttentionConfig | None = None,
) -> "CLIP":
    """Load pretrained CLIP model.

    Args:
        cls: CLIP class.
        model_name_or_path (str): Model path or ID.
        use_pytorch (bool): Load from PyTorch.
        rngs (rnglib.Rngs | None): RNG state. If None, initializes to nnx.Rngs(0).
        dtype (DTypeLike): Computation dtype.
        param_dtype (DTypeLike): Parameter dtype.
        mesh (Mesh | None): Device mesh.
        use_gradient_checkpointing (bool): Enable gradient checkpointing.
        splash_attention_config (SplashAttentionConfig | None): Configuration for TPU splash attention. Defaults to None.

    Returns:
        CLIP: Loaded model.
    """
    if rngs is None:
        rngs = nnx.Rngs(0)
    params_fstate, config_dict = load_params_and_config(model_name_or_path, use_pytorch)

    if not config_dict:
        config_dict = _infer_config_from_params(params_fstate, use_pytorch)

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
        splash_attention_config=splash_attention_config,
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
        mesh=mesh,
    )

    mapping = _build_param_mapping(
        config_dict,
        component="both",
        text_config=text_config,
        vision_config=vision_config,
    )
    transform_fn = _create_transform_fn(config_dict)
    load_and_apply_params(model, params_fstate, mapping, transform_fn, param_dtype)

    model._original_config = config_dict
    return model
