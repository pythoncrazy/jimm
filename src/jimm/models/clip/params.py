import json
import os
from typing import Any, Set

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib
from jax.sharding import Mesh
from jaxtyping import DTypeLike
from safetensors.flax import save_file as save_safetensors

from jimm.common.utils import convert_state_to_hf_format, load_params_and_config
from jimm.models import CLIP, CLIPVisionModel


def _create_config(model: "CLIP") -> dict[str, Any]:
    """Create HuggingFace-compatible config dictionary.

    Args:
        model (CLIP): The CLIP model instance.

    Returns:
        dict[str, Any]: Configuration dictionary in HuggingFace format.
    """
    return {
        "model_type": "clip",
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


def save_pretrained(model: "CLIP", save_directory: str) -> None:
    """Save the CLIP model weights and config in HuggingFace format.

    Args:
        model (CLIP): The CLIP model instance to save.
        save_directory (str): Directory path where the model will be saved.
    """
    _SPECIAL_MAPPINGS = {
        "ln_final.weight": "text_model.final_layer_norm.weight",
        "ln_final.bias": "text_model.final_layer_norm.bias",
        "vision_model.encoder.ln_pre.weight": "vision_model.pre_layrnorm.weight",
        "vision_model.encoder.ln_pre.bias": "vision_model.pre_layrnorm.bias",
        "vision_model.encoder.ln_post.weight": "vision_model.post_layernorm.weight",
        "vision_model.encoder.ln_post.bias": "vision_model.post_layernorm.bias",
        "vision_model.encoder.cls_token": "vision_model.embeddings.class_embedding",
        "vision_model.encoder.position_embeddings": "vision_model.embeddings.position_embedding.weight",
        "vision_model.encoder.patch_embeddings.weight": "vision_model.embeddings.patch_embedding.weight",
        "positional_embedding": "text_model.embeddings.position_embedding.weight",
        "vision_model.vision_position_ids": "vision_model.embeddings.position_ids",
        "token_embedding.embedding": "text_model.embeddings.token_embedding.weight",
        "text_projection.weight": "text_projection.weight",
        "vision_model.visual_projection.weight": "visual_projection.weight",
    }
    _SPECIAL_RENAMINGS = {
        "text_model.layers": "text_model.encoder.layers",
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
    if jax.process_index() == 0:
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    _, state = nnx.split(model)
    hf_state = convert_state_to_hf_format(nnx.to_pure_dict(state), _SPECIAL_MAPPINGS, _SPECIAL_RENAMINGS)
    save_safetensors(hf_state, os.path.join(save_directory, "model.safetensors"))


def load_vision_from_pretrained(
    cls,
    model_name_or_path: str,
    use_pytorch: bool = False,
    mesh: Mesh | None = None,
    dtype: DTypeLike = jnp.float32,
    param_dtype: DTypeLike = jnp.float32,
    use_gradient_checkpointing: bool = False,
    rngs: rnglib.Rngs = nnx.Rngs(0),
) -> "CLIPVisionModel":
    """Load a pretrained vision encoder from a CLIP checkpoint.

    Args:
        cls: The CLIPVisionModel class.
        model_name_or_path (str): Path to local weights or HuggingFace model ID.
        use_pytorch (bool): Whether to load from PyTorch weights. Defaults to False.
        mesh (Mesh | None): Optional device mesh for parameter sharding. Defaults to None.
        dtype (DTypeLike): Data type for computations. Defaults to jnp.float32.
        param_dtype (DTypeLike): Data type for parameters. Defaults to jnp.float32.
        use_gradient_checkpointing (bool): Whether to use gradient checkpointing. Defaults to False.
        rngs (rnglib.Rngs): Random number generator keys. Defaults to nnx.Rngs(0).

    Returns:
        CLIPVisionModel: Pretrained CLIP vision model
    """
    params_fstate, config_dict = load_params_and_config(model_name_or_path, use_pytorch)

    config: dict[str, Any] = config_dict

    if config == {}:
        if not use_pytorch:
            text_hidden_size = params_fstate["text_model.embeddings.token_embedding.weight"].shape[1]

            vision_hidden_size = params_fstate["vision_model.embeddings.class_embedding"].shape[0]
            vision_patch_size = params_fstate["vision_model.embeddings.patch_embedding.weight"].shape[2]
            vision_image_size = int((params_fstate["vision_model.embeddings.position_embedding.weight"].shape[0] - 1) ** 0.5) * vision_patch_size

            vision_num_layers = 0
            for k_param in params_fstate:
                if k_param.startswith("vision_model.encoder.layers.") and k_param.endswith(".self_attn.q_proj.weight"):
                    layer_idx = int(k_param.split(".")[3])
                    vision_num_layers = max(vision_num_layers, layer_idx + 1)

            config = {
                "text_config": {
                    "hidden_size": text_hidden_size,
                },
                "vision_config": {
                    "hidden_size": vision_hidden_size,
                    "num_attention_heads": vision_hidden_size // 64,
                    "num_hidden_layers": vision_num_layers,
                    "image_size": vision_image_size,
                    "patch_size": vision_patch_size,
                },
            }
        else:
            raise ValueError(f"Configuration could not be loaded for PyTorch model {model_name_or_path}")

    text_config = config["text_config"]
    vision_config = config["vision_config"]

    vision_model = cls(
        image_resolution=vision_config["image_size"],
        vision_layers=vision_config["num_hidden_layers"],
        vision_width=vision_config["hidden_size"],
        vision_patch_size=vision_config["patch_size"],
        transformer_width=text_config["hidden_size"],
        use_gradient_checkpointing=use_gradient_checkpointing,
        mesh=mesh,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
    )

    flax_model_params_fstate = dict(nnx.to_flat_state(nnx.state(vision_model, nnx.Param)))

    vision_mapping_list = [
        (("encoder", "cls_token"), ("vision_model", "embeddings", "class_embedding")),
        (("encoder", "position_embeddings"), ("vision_model", "embeddings", "position_embedding", "weight")),
        (("encoder", "patch_embeddings", "kernel"), ("vision_model", "embeddings", "patch_embedding", "weight")),
        (("encoder", "ln_pre", "scale"), ("vision_model", "pre_layrnorm", "weight")),
        (("encoder", "ln_pre", "bias"), ("vision_model", "pre_layrnorm", "bias")),
        (("encoder", "ln_post", "scale"), ("vision_model", "post_layernorm", "weight")),
        (("encoder", "ln_post", "bias"), ("vision_model", "post_layernorm", "bias")),
        (("visual_projection", "kernel"), ("visual_projection", "weight")),
    ]

    for i in range(vision_config["num_hidden_layers"]):
        flax_base = ("encoder", "encoder", f"layers_{i}")
        hf_base = ("vision_model", "encoder", "layers", str(i))

        vision_mapping_list.extend(
            [
                (flax_base + ("attn", "query", "kernel"), hf_base + ("self_attn", "q_proj", "weight")),
                (flax_base + ("attn", "query", "bias"), hf_base + ("self_attn", "q_proj", "bias")),
                (flax_base + ("attn", "key", "kernel"), hf_base + ("self_attn", "k_proj", "weight")),
                (flax_base + ("attn", "key", "bias"), hf_base + ("self_attn", "k_proj", "bias")),
                (flax_base + ("attn", "value", "kernel"), hf_base + ("self_attn", "v_proj", "weight")),
                (flax_base + ("attn", "value", "bias"), hf_base + ("self_attn", "v_proj", "bias")),
                (flax_base + ("attn", "out", "kernel"), hf_base + ("self_attn", "out_proj", "weight")),
                (flax_base + ("attn", "out", "bias"), hf_base + ("self_attn", "out_proj", "bias")),
                (flax_base + ("norm1", "scale"), hf_base + ("layer_norm1", "weight")),
                (flax_base + ("norm1", "bias"), hf_base + ("layer_norm1", "bias")),
                (flax_base + ("norm2", "scale"), hf_base + ("layer_norm2", "weight")),
                (flax_base + ("norm2", "bias"), hf_base + ("layer_norm2", "bias")),
                (flax_base + ("mlp", "layers", 0, "kernel"), hf_base + ("mlp", "fc1", "weight")),
                (flax_base + ("mlp", "layers", 0, "bias"), hf_base + ("mlp", "fc1", "bias")),
                (flax_base + ("mlp", "layers", 3, "kernel"), hf_base + ("mlp", "fc2", "weight")),
                (flax_base + ("mlp", "layers", 3, "bias"), hf_base + ("mlp", "fc2", "bias")),
            ]
        )

    vision_projection_mapping = [
        (("visual_projection", "kernel"), ("visual_projection", "weight")),
    ]
    params_name_mapping = dict(vision_mapping_list + vision_projection_mapping)
    nonvisited = set(flax_model_params_fstate.keys())

    for flax_dst_key_tuple, hf_src_key_tuple in params_name_mapping.items():
        hf_src_key_as_string = ".".join(hf_src_key_tuple)

        nonvisited.discard(flax_dst_key_tuple)
        src_value = params_fstate[hf_src_key_as_string]
        dst_value_obj = flax_model_params_fstate[flax_dst_key_tuple]

        if flax_dst_key_tuple == ("encoder", "patch_embeddings", "kernel"):
            src_value = jnp.transpose(src_value, (2, 3, 1, 0))
        elif flax_dst_key_tuple == ("encoder", "cls_token"):
            src_value = src_value.reshape(1, 1, -1)
        elif flax_dst_key_tuple == ("encoder", "position_embeddings"):
            src_value = src_value.reshape(1, src_value.shape[0], src_value.shape[1])
        elif hf_src_key_tuple[-1] == "weight" and hf_src_key_tuple[-2] in ("q_proj", "k_proj", "v_proj"):
            src_value = jnp.transpose(src_value, (1, 0))
            num_heads = vision_config["hidden_size"] // 64
            hidden_size = vision_config["hidden_size"]
            head_dim = hidden_size // num_heads
            src_value = src_value.reshape((hidden_size, num_heads, head_dim))
        elif hf_src_key_tuple[-1] == "bias" and hf_src_key_tuple[-2] in ("q_proj", "k_proj", "v_proj"):
            num_heads = vision_config["hidden_size"] // 64
            hidden_size = vision_config["hidden_size"]
            head_dim = hidden_size // num_heads
            src_value = src_value.reshape((num_heads, head_dim))
        elif hf_src_key_tuple[-2:] == ("out_proj", "weight"):
            src_value = jnp.transpose(src_value, (1, 0))
            num_heads = vision_config["hidden_size"] // 64
            hidden_size = vision_config["hidden_size"]
            head_dim = hidden_size // num_heads
            src_value = src_value.reshape((num_heads, head_dim, hidden_size))
        elif hf_src_key_tuple[-1] == "weight" and src_value.ndim == 2:
            src_value = jnp.transpose(src_value, (1, 0))

        if src_value.shape != dst_value_obj.value.shape:
            raise ValueError(f"Shape mismatch for {flax_dst_key_tuple} vs {hf_src_key_as_string}: {dst_value_obj.value.shape} (expected) != {src_value.shape} (actual)")

        src_value = src_value.astype(param_dtype)
        dst_value_obj.value = src_value

    nnx.update(vision_model, nnx.from_flat_state(flax_model_params_fstate))

    known_buffer_keys = {("encoder", "vision_position_ids"), ("visual_projection",)}
    unexpected_nonvisited = nonvisited - known_buffer_keys
    if unexpected_nonvisited:
        print(f"Warning: Some CLIPVisionModel parameters were not loaded: {sorted(list(unexpected_nonvisited))}")

    if ("visual_projection", "kernel") not in flax_model_params_fstate and "visual_projection.weight" in params_fstate:
        vision_model.visual_projection.kernel.value = jnp.transpose(params_fstate["visual_projection.weight"], (1, 0)).astype(param_dtype)

    return vision_model


def load_from_pretrained(
    cls,
    model_name_or_path: str,
    use_pytorch: bool = False,
    mesh: Mesh | None = None,
    dtype: DTypeLike = jnp.float32,
    param_dtype: DTypeLike = jnp.float32,
    use_gradient_checkpointing: bool = False,
    rngs: rnglib.Rngs = nnx.Rngs(0),
) -> "CLIP":
    """Load a pretrained CLIP model from a local path or HuggingFace Hub.

    Args:
        cls: The CLIP class.
        model_name_or_path (str): Path to local weights or HuggingFace model ID.
        use_pytorch (bool): Whether to load from PyTorch weights. Defaults to False.
        mesh (Mesh | None): Optional device mesh for parameter sharding. Defaults to None.
        dtype (DTypeLike): Data type for computations. Defaults to jnp.float32.
        param_dtype (DTypeLike): Data type for parameters. Defaults to jnp.float32.
        use_gradient_checkpointing (bool): Whether to use gradient checkpointing. Defaults to False.
        rngs (rnglib.Rngs): Random number generator keys. Defaults to nnx.Rngs(0).

    Returns:
        CLIP: Pretrained CLIP model
    """

    params_fstate, config_dict = load_params_and_config(model_name_or_path, use_pytorch)

    config: dict[str, Any] = config_dict

    if config == {}:
        if not use_pytorch:
            text_hidden_size = params_fstate["text_model.embeddings.token_embedding.weight"].shape[1]
            text_max_pos_embed = params_fstate["text_model.embeddings.position_embedding.weight"].shape[0]
            text_vocab_size = params_fstate["text_model.embeddings.token_embedding.weight"].shape[0]

            text_num_layers = 0
            for k_param in params_fstate:
                if k_param.startswith("text_model.encoder.layers.") and k_param.endswith(".self_attn.q_proj.weight"):
                    layer_idx = int(k_param.split(".")[3])
                    text_num_layers = max(text_num_layers, layer_idx + 1)

            vision_hidden_size = params_fstate["vision_model.embeddings.class_embedding"].shape[0]
            vision_patch_size = params_fstate["vision_model.embeddings.patch_embedding.weight"].shape[2]
            vision_image_size = int((params_fstate["vision_model.embeddings.position_embedding.weight"].shape[0] - 1) ** 0.5) * vision_patch_size

            vision_num_layers = 0
            for k_param in params_fstate:
                if k_param.startswith("vision_model.encoder.layers.") and k_param.endswith(".self_attn.q_proj.weight"):
                    layer_idx = int(k_param.split(".")[3])
                    vision_num_layers = max(vision_num_layers, layer_idx + 1)

            config = {
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
        else:
            raise ValueError(f"Configuration could not be loaded for PyTorch model {model_name_or_path}")

    text_config = config["text_config"]
    vision_config = config["vision_config"]

    model = cls(
        image_resolution=vision_config["image_size"],
        vision_layers=vision_config["num_hidden_layers"],
        vision_width=vision_config["hidden_size"],
        vision_patch_size=vision_config["patch_size"],
        context_length=text_config["max_position_embeddings"],
        vocab_size=text_config["vocab_size"],
        transformer_width=text_config["hidden_size"],
        transformer_heads=text_config["num_attention_heads"],
        transformer_layers=text_config["num_hidden_layers"],
        use_gradient_checkpointing=use_gradient_checkpointing,
        mesh=mesh,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
    )

    flax_model_params_fstate = dict(nnx.to_flat_state(nnx.state(model, nnx.Param)))

    mapping_list = [
        (("logit_scale",), ("logit_scale",)),
        (("positional_embedding",), ("text_model", "embeddings", "position_embedding", "weight")),
        (("token_embedding", "embedding"), ("text_model", "embeddings", "token_embedding", "weight")),
        (("ln_final", "scale"), ("text_model", "final_layer_norm", "weight")),
        (("ln_final", "bias"), ("text_model", "final_layer_norm", "bias")),
        (("text_projection", "kernel"), ("text_projection", "weight")),
        (("vision_model", "encoder", "cls_token"), ("vision_model", "embeddings", "class_embedding")),
        (("vision_model", "encoder", "position_embeddings"), ("vision_model", "embeddings", "position_embedding", "weight")),
        (("vision_model", "encoder", "patch_embeddings", "kernel"), ("vision_model", "embeddings", "patch_embedding", "weight")),
        (("vision_model", "encoder", "ln_pre", "scale"), ("vision_model", "pre_layrnorm", "weight")),
        (("vision_model", "encoder", "ln_pre", "bias"), ("vision_model", "pre_layrnorm", "bias")),
        (("vision_model", "encoder", "ln_post", "scale"), ("vision_model", "post_layernorm", "weight")),
        (("vision_model", "encoder", "ln_post", "bias"), ("vision_model", "post_layernorm", "bias")),
        (("vision_model", "visual_projection", "kernel"), ("visual_projection", "weight")),
    ]

    for i in range(text_config["num_hidden_layers"]):
        flax_base = ("text_model", f"layers_{i}")
        hf_base = ("text_model", "encoder", "layers", str(i))

        mapping_list.extend(
            [
                (flax_base + ("attn", "query", "kernel"), hf_base + ("self_attn", "q_proj", "weight")),
                (flax_base + ("attn", "query", "bias"), hf_base + ("self_attn", "q_proj", "bias")),
                (flax_base + ("attn", "key", "kernel"), hf_base + ("self_attn", "k_proj", "weight")),
                (flax_base + ("attn", "key", "bias"), hf_base + ("self_attn", "k_proj", "bias")),
                (flax_base + ("attn", "value", "kernel"), hf_base + ("self_attn", "v_proj", "weight")),
                (flax_base + ("attn", "value", "bias"), hf_base + ("self_attn", "v_proj", "bias")),
                (flax_base + ("attn", "out", "kernel"), hf_base + ("self_attn", "out_proj", "weight")),
                (flax_base + ("attn", "out", "bias"), hf_base + ("self_attn", "out_proj", "bias")),
                (flax_base + ("norm1", "scale"), hf_base + ("layer_norm1", "weight")),
                (flax_base + ("norm1", "bias"), hf_base + ("layer_norm1", "bias")),
                (flax_base + ("norm2", "scale"), hf_base + ("layer_norm2", "weight")),
                (flax_base + ("norm2", "bias"), hf_base + ("layer_norm2", "bias")),
                (flax_base + ("mlp", "layers", 0, "kernel"), hf_base + ("mlp", "fc1", "weight")),
                (flax_base + ("mlp", "layers", 0, "bias"), hf_base + ("mlp", "fc1", "bias")),
                (flax_base + ("mlp", "layers", 3, "kernel"), hf_base + ("mlp", "fc2", "weight")),
                (flax_base + ("mlp", "layers", 3, "bias"), hf_base + ("mlp", "fc2", "bias")),
            ]
        )

    for i in range(vision_config["num_hidden_layers"]):
        flax_base = ("vision_model", "encoder", "encoder", f"layers_{i}")
        hf_base = ("vision_model", "encoder", "layers", str(i))

        mapping_list.extend(
            [
                (flax_base + ("attn", "query", "kernel"), hf_base + ("self_attn", "q_proj", "weight")),
                (flax_base + ("attn", "query", "bias"), hf_base + ("self_attn", "q_proj", "bias")),
                (flax_base + ("attn", "key", "kernel"), hf_base + ("self_attn", "k_proj", "weight")),
                (flax_base + ("attn", "key", "bias"), hf_base + ("self_attn", "k_proj", "bias")),
                (flax_base + ("attn", "value", "kernel"), hf_base + ("self_attn", "v_proj", "weight")),
                (flax_base + ("attn", "value", "bias"), hf_base + ("self_attn", "v_proj", "bias")),
                (flax_base + ("attn", "out", "kernel"), hf_base + ("self_attn", "out_proj", "weight")),
                (flax_base + ("attn", "out", "bias"), hf_base + ("self_attn", "out_proj", "bias")),
                (flax_base + ("norm1", "scale"), hf_base + ("layer_norm1", "weight")),
                (flax_base + ("norm1", "bias"), hf_base + ("layer_norm1", "bias")),
                (flax_base + ("norm2", "scale"), hf_base + ("layer_norm2", "weight")),
                (flax_base + ("norm2", "bias"), hf_base + ("layer_norm2", "bias")),
                (flax_base + ("mlp", "layers", 0, "kernel"), hf_base + ("mlp", "fc1", "weight")),
                (flax_base + ("mlp", "layers", 0, "bias"), hf_base + ("mlp", "fc1", "bias")),
                (flax_base + ("mlp", "layers", 3, "kernel"), hf_base + ("mlp", "fc2", "weight")),
                (flax_base + ("mlp", "layers", 3, "bias"), hf_base + ("mlp", "fc2", "bias")),
            ]
        )

    params_name_mapping = dict(mapping_list)
    nonvisited = set(flax_model_params_fstate.keys())

    hf_checkpoint_keys: Set[str] = set(params_fstate.keys())
    used_hf_keys: Set[str] = set()

    for flax_dst_key_tuple, hf_src_key_tuple in params_name_mapping.items():
        hf_src_key_as_string = ".".join(hf_src_key_tuple)

        used_hf_keys.add(hf_src_key_as_string)
        nonvisited.discard(flax_dst_key_tuple)
        src_value = params_fstate[hf_src_key_as_string]
        dst_value_obj = flax_model_params_fstate[flax_dst_key_tuple]

        if flax_dst_key_tuple[0] == "vision_model":
            if flax_dst_key_tuple == ("vision_model", "encoder", "patch_embeddings", "kernel"):
                src_value = jnp.transpose(src_value, (2, 3, 1, 0))
            elif flax_dst_key_tuple == ("vision_model", "encoder", "cls_token"):
                src_value = src_value.reshape(1, 1, -1)
            elif flax_dst_key_tuple == ("vision_model", "encoder", "position_embeddings"):
                src_value = src_value.reshape(1, src_value.shape[0], src_value.shape[1])
            elif hf_src_key_tuple[-1] == "weight" and hf_src_key_tuple[-2] in ("q_proj", "k_proj", "v_proj"):
                src_value = jnp.transpose(src_value, (1, 0))
                num_heads = vision_config["hidden_size"] // 64
                hidden_size = vision_config["hidden_size"]
                head_dim = hidden_size // num_heads
                src_value = src_value.reshape((hidden_size, num_heads, head_dim))
            elif hf_src_key_tuple[-1] == "bias" and hf_src_key_tuple[-2] in ("q_proj", "k_proj", "v_proj"):
                num_heads = vision_config["hidden_size"] // 64
                hidden_size = vision_config["hidden_size"]
                head_dim = hidden_size // num_heads
                src_value = src_value.reshape((num_heads, head_dim))
            elif hf_src_key_tuple[-2:] == ("out_proj", "weight"):
                src_value = jnp.transpose(src_value, (1, 0))
                num_heads = vision_config["hidden_size"] // 64
                hidden_size = vision_config["hidden_size"]
                head_dim = hidden_size // num_heads
                src_value = src_value.reshape((num_heads, head_dim, hidden_size))
            elif hf_src_key_tuple[-1] == "weight" and src_value.ndim == 2:
                src_value = jnp.transpose(src_value, (1, 0))
        elif hf_src_key_tuple[-1] == "weight" and hf_src_key_tuple[-2] in ("q_proj", "k_proj", "v_proj"):
            src_value = jnp.transpose(src_value, (1, 0))
            num_heads = text_config["num_attention_heads"]
            hidden_size = text_config["hidden_size"]
            head_dim = hidden_size // num_heads
            src_value = src_value.reshape((hidden_size, num_heads, head_dim))
        elif hf_src_key_tuple[-1] == "bias" and hf_src_key_tuple[-2] in ("q_proj", "k_proj", "v_proj"):
            num_heads = text_config["num_attention_heads"]
            hidden_size = text_config["hidden_size"]
            head_dim = hidden_size // num_heads
            src_value = src_value.reshape((num_heads, head_dim))
        elif hf_src_key_tuple[-2:] == ("out_proj", "weight"):
            src_value = jnp.transpose(src_value, (1, 0))
            num_heads = text_config["num_attention_heads"]
            hidden_size = text_config["hidden_size"]
            head_dim = hidden_size // num_heads
            src_value = src_value.reshape((num_heads, head_dim, hidden_size))
        elif flax_dst_key_tuple == ("token_embedding", "embedding") or flax_dst_key_tuple == ("positional_embedding",):
            pass
        elif hf_src_key_tuple[-1] == "weight" and src_value.ndim == 2:
            src_value = jnp.transpose(src_value, (1, 0))

        if src_value.shape != dst_value_obj.value.shape:
            raise ValueError(f"Shape mismatch for {flax_dst_key_tuple} (Flax) vs {hf_src_key_as_string} (HF): {dst_value_obj.value.shape} (expected) != {src_value.shape} (actual)")

        src_value = src_value.astype(param_dtype)
        dst_value_obj.value = src_value

    nnx.update(model, nnx.from_flat_state(flax_model_params_fstate))

    known_buffer_keys = {
        ("vision_model", "encoder", "vision_position_ids"),
    }
    unexpected_nonvisited = nonvisited - known_buffer_keys
    if unexpected_nonvisited:
        print(f"Warning: Some CLIP parameters were not loaded: {sorted(list(unexpected_nonvisited))}")

    leftover_hf_keys = hf_checkpoint_keys - used_hf_keys

    known_unused_hf_buffer_keys = {
        "text_model.embeddings.position_ids",
        "vision_model.embeddings.position_ids",
        "vision_model.encoder.vision_position_ids",
    }
    unexpected_leftover_hf_keys = leftover_hf_keys - known_unused_hf_buffer_keys

    assert len(unexpected_leftover_hf_keys) == 0, f"Some unexpected HuggingFace checkpoint parameters were not used: {sorted(list(unexpected_leftover_hf_keys))}"

    model._original_config = config
    return model
