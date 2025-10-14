import json
import os
from typing import Any, Set, TYPE_CHECKING

import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib
from jax.sharding import Mesh
from jaxtyping import DTypeLike
from safetensors.flax import save_file as save_safetensors

from jimm.common.utils import convert_state_to_hf_format, load_params_and_config

if TYPE_CHECKING:
    from jimm.models import SigLIP, SigLIPVisionModel


def _create_config(model: "SigLIP") -> dict[str, Any]:
    """Create HuggingFace-compatible config dictionary.

    Args:
        model (SigLIP): The SigLIP model instance.

    Returns:
        dict[str, Any]: Configuration dictionary in HuggingFace format.
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
    """Save the SigLIP model weights and config in HuggingFace format.

    Args:
        model (SigLIP): The SigLIP model instance to save.
        save_directory (str): Directory path where the model will be saved.
    """
    _SPECIAL_MAPPINGS = {
        "ln_final.weight": "text_model.final_layer_norm.weight",
        "ln_final.bias": "text_model.final_layer_norm.bias",
        "vision_model.encoder.ln_pre.weight": "vision_model.pre_layrnorm.weight",
        "vision_model.encoder.ln_pre.bias": "vision_model.pre_layrnorm.bias",
        "vision_model.encoder.ln_post.weight": "vision_model.post_layernorm.weight",
        "vision_model.encoder.ln_post.bias": "vision_model.post_layernorm.bias",
        "vision_model.encoder.position_embeddings": "vision_model.embeddings.position_embedding.weight",
        "vision_model.encoder.patch_embeddings.weight": "vision_model.embeddings.patch_embedding.weight",
        "vision_model.encoder.patch_embeddings.bias": "vision_model.embeddings.patch_embedding.bias",
        "positional_embedding": "text_model.embeddings.position_embedding.weight",
        "text_position_ids": "text_model.embeddings.position_ids",
        "token_embedding.embedding": "text_model.embeddings.token_embedding.weight",
        "text_projection.weight": "text_model.head.weight",
        "text_projection.bias": "text_model.head.bias",
        "visual_projection.weight": "visual_projection.weight",
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


def load_vision_from_pretrained(
    cls,
    model_name_or_path: str,
    use_pytorch: bool = False,
    mesh: Mesh | None = None,
    dtype: DTypeLike = jnp.float32,
    param_dtype: DTypeLike = jnp.float32,
    use_gradient_checkpointing: bool = False,
    rngs: rnglib.Rngs = nnx.Rngs(0),
) -> "SigLIPVisionModel":
    """Load a pretrained vision encoder from a SigLIP checkpoint.

    Args:
        cls: The SigLIPVisionModel class.
        model_name_or_path (str): Path to local weights or HuggingFace model ID.
        use_pytorch (bool): Whether to load from PyTorch weights. Defaults to False.
        mesh (Mesh | None): Optional device mesh for parameter sharding. Defaults to None.
        dtype (DTypeLike): Data type for computations. Defaults to jnp.float32.
        param_dtype (DTypeLike): Data type for parameters. Defaults to jnp.float32.
        use_gradient_checkpointing (bool): Whether to use gradient checkpointing. Defaults to False.
        rngs (rnglib.Rngs): Random number generator keys. Defaults to nnx.Rngs(0).

    Returns:
        SigLIPVisionModel: Pretrained SigLIP vision model
    """
    params_fstate, config_dict = load_params_and_config(model_name_or_path, use_pytorch)

    vision_patch_size = params_fstate["vision_model.embeddings.patch_embedding.weight"].shape[3]
    vision_width = params_fstate["vision_model.embeddings.patch_embedding.bias"].shape[0]
    vision_num_layers = 0
    for k in params_fstate:
        if k.startswith("vision_model.encoder.layers.") and k.endswith(".mlp.fc2.bias"):
            vision_num_layers = max(vision_num_layers, int(k.split(".")[3]) + 1)

    vision_model = cls(
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

    flax_model_params_fstate = dict(nnx.to_flat_state(nnx.state(vision_model, nnx.Param)))

    vision_mapping_list = [
        (("encoder", "patch_embeddings", "kernel"), ("vision_model", "embeddings", "patch_embedding", "weight")),
        (("encoder", "patch_embeddings", "bias"), ("vision_model", "embeddings", "patch_embedding", "bias")),
        (("encoder", "position_embeddings"), ("vision_model", "embeddings", "position_embedding", "weight")),
        (("encoder", "ln_post", "scale"), ("vision_model", "post_layernorm", "weight")),
        (("encoder", "ln_post", "bias"), ("vision_model", "post_layernorm", "bias")),
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

    vision_heads = vision_width // 64

    for i in range(vision_num_layers):
        flax_base = ("encoder", "encoder", "layers", i)
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

    params_name_mapping = dict(vision_mapping_list)
    nonvisited = set(flax_model_params_fstate.keys())
    used_hf_keys: Set[str] = set()

    for flax_dst_key_tuple, hf_src_key_tuple in params_name_mapping.items():
        hf_src_key_as_string = ".".join(hf_src_key_tuple)

        nonvisited.discard(flax_dst_key_tuple)
        used_hf_keys.add(hf_src_key_as_string)
        src_value = params_fstate[hf_src_key_as_string]
        dst_value_obj = flax_model_params_fstate[flax_dst_key_tuple]

        if flax_dst_key_tuple == ("encoder", "patch_embeddings", "kernel"):
            src_value = jnp.transpose(src_value, (2, 3, 1, 0))
        elif flax_dst_key_tuple == ("encoder", "position_embeddings"):
            src_value = src_value.reshape(1, src_value.shape[0], src_value.shape[1])
        elif hf_src_key_tuple[-1] == "weight" and hf_src_key_tuple[-2] in ("q_proj", "k_proj", "v_proj"):
            src_value = jnp.transpose(src_value, (1, 0))
            num_heads = vision_heads
            head_dim = vision_width // num_heads
            src_value = src_value.reshape((vision_width, num_heads, head_dim))
        elif hf_src_key_tuple[-1] == "bias" and hf_src_key_tuple[-2] in ("q_proj", "k_proj", "v_proj"):
            num_heads = vision_heads
            head_dim = vision_width // num_heads
            src_value = src_value.reshape((num_heads, head_dim))
        elif hf_src_key_tuple[-2:] == ("out_proj", "weight"):
            src_value = jnp.transpose(src_value, (1, 0))
            num_heads = vision_heads
            head_dim = vision_width // num_heads
            src_value = src_value.reshape((num_heads, head_dim, vision_width))
        elif hf_src_key_tuple[-1] == "in_proj_weight":
            num_heads = vision_heads
            head_dim = vision_width // num_heads
            q_w, k_w, v_w = jnp.split(src_value, 3, axis=0)
            w_map = {"query": q_w, "key": k_w, "value": v_w}
            src_value = jnp.transpose(w_map[flax_dst_key_tuple[-2]], (1, 0)).reshape(vision_width, num_heads, head_dim)
        elif hf_src_key_tuple[-1] == "in_proj_bias":
            num_heads = vision_heads
            head_dim = vision_width // num_heads
            q_b, k_b, v_b = jnp.split(src_value, 3, axis=0)
            b_map = {"query": q_b, "key": k_b, "value": v_b}
            src_value = b_map[flax_dst_key_tuple[-2]].reshape(num_heads, head_dim)
        elif hf_src_key_tuple[-1] == "weight" and src_value.ndim == 2:
            src_value = jnp.transpose(src_value, (1, 0))

        if src_value.shape != dst_value_obj.value.shape:
            raise ValueError(f"Shape mismatch for {flax_dst_key_tuple} vs {hf_src_key_as_string}: {dst_value_obj.value.shape} (expected) != {src_value.shape} (actual)")

        src_value = src_value.astype(param_dtype)
        dst_value_obj.value = src_value

    nnx.update(vision_model, nnx.from_flat_state(flax_model_params_fstate))
    known_buffer_keys = {("encoder", "vision_position_ids")}
    unexpected_nonvisited = nonvisited - known_buffer_keys
    if unexpected_nonvisited:
        print(f"Warning: Some {cls.__name__} parameters were not loaded: {sorted(list(unexpected_nonvisited))}")

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
) -> "SigLIP":
    """Load a pretrained SigLIP model from a local path or HuggingFace Hub.

    Args:
        cls: The SigLIP class.
        model_name_or_path (str): Path to local weights or HuggingFace model ID.
        use_pytorch (bool): Whether to load from PyTorch weights. Defaults to False.
        mesh (Mesh | None): Optional device mesh for parameter sharding. Defaults to None.
        dtype (DTypeLike): Data type for computations. Defaults to jnp.float32.
        param_dtype (DTypeLike): Data type for parameters. Defaults to jnp.float32.
        use_gradient_checkpointing (bool): Whether to use gradient checkpointing. Defaults to False.
        rngs (rnglib.Rngs): Random number generator keys. Defaults to nnx.Rngs(0).

    Returns:
        SigLIP: Pretrained SigLIP model
    """
    params_fstate, config_dict = load_params_and_config(model_name_or_path, use_pytorch)
    config: dict[str, Any] = config_dict

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

    model = cls(
        image_resolution=config["vision_config"]["image_size"],
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

    flax_model_params_fstate = dict(nnx.to_flat_state(nnx.state(model, nnx.Param)))
    nonvisited = set(flax_model_params_fstate.keys())
    used_hf_keys: Set[str] = set()

    mapping_list = [
        (("logit_scale",), ("logit_scale",)),
        (("logit_bias",), ("logit_bias",)),
        (("positional_embedding",), ("text_model", "embeddings", "position_embedding", "weight")),
        (("token_embedding", "embedding"), ("text_model", "embeddings", "token_embedding", "weight")),
        (("ln_final", "scale"), ("text_model", "final_layer_norm", "weight")),
        (("ln_final", "bias"), ("text_model", "final_layer_norm", "bias")),
        (("text_projection", "kernel"), ("text_model", "head", "weight")),
        (("text_projection", "bias"), ("text_model", "head", "bias")),
        (("vision_model", "encoder", "patch_embeddings", "kernel"), ("vision_model", "embeddings", "patch_embedding", "weight")),
        (("vision_model", "encoder", "patch_embeddings", "bias"), ("vision_model", "embeddings", "patch_embedding", "bias")),
        (("vision_model", "encoder", "position_embeddings"), ("vision_model", "embeddings", "position_embedding", "weight")),
        (("vision_model", "encoder", "ln_post", "scale"), ("vision_model", "post_layernorm", "weight")),
        (("vision_model", "encoder", "ln_post", "bias"), ("vision_model", "post_layernorm", "bias")),
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

    for i in range(text_num_layers):
        flax_base = ("text_model", "layers", i)
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

    for i in range(vision_num_layers):
        flax_base = ("vision_model", "encoder", "encoder", "layers", i)
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

    for flax_dst_key_tuple, hf_src_key_tuple in params_name_mapping.items():
        hf_src_key_as_string = ".".join(hf_src_key_tuple)
        nonvisited.discard(flax_dst_key_tuple)
        used_hf_keys.add(hf_src_key_as_string)
        src_value = params_fstate[hf_src_key_as_string]
        dst_value_obj = flax_model_params_fstate[flax_dst_key_tuple]

        if flax_dst_key_tuple == ("vision_model", "encoder", "patch_embeddings", "kernel"):
            src_value = jnp.transpose(src_value, (2, 3, 1, 0))
        elif flax_dst_key_tuple == ("vision_model", "encoder", "position_embeddings"):
            src_value = src_value.reshape(1, src_value.shape[0], src_value.shape[1])
        elif flax_dst_key_tuple in [("logit_scale",), ("logit_bias",)]:
            src_value = jnp.squeeze(src_value)
        elif hf_src_key_tuple[-1] == "weight" and hf_src_key_tuple[-2] in ("q_proj", "k_proj", "v_proj"):
            src_value = jnp.transpose(src_value, (1, 0))
            if "text_model" in hf_src_key_as_string:
                num_heads = model.transformer_heads
                head_dim = model.transformer_width // num_heads
                src_value = src_value.reshape((model.transformer_width, num_heads, head_dim))
            else:
                num_heads = model.vision_heads
                head_dim = vision_width // num_heads
                src_value = src_value.reshape((vision_width, num_heads, head_dim))
        elif hf_src_key_tuple[-1] == "bias" and hf_src_key_tuple[-2] in ("q_proj", "k_proj", "v_proj"):
            if "text_model" in hf_src_key_as_string:
                num_heads = model.transformer_heads
                head_dim = model.transformer_width // num_heads
            else:
                num_heads = model.vision_heads
                head_dim = vision_width // num_heads
            src_value = src_value.reshape((num_heads, head_dim))
        elif hf_src_key_tuple[-2:] == ("out_proj", "weight"):
            src_value = jnp.transpose(src_value, (1, 0))
            if "text_model" in hf_src_key_as_string:
                num_heads = model.transformer_heads
                head_dim = model.transformer_width // num_heads
                src_value = src_value.reshape((num_heads, head_dim, model.transformer_width))
            else:
                num_heads = model.vision_heads
                head_dim = vision_width // num_heads
                src_value = src_value.reshape((num_heads, head_dim, vision_width))
        elif hf_src_key_tuple[-1] == "in_proj_weight":
            num_heads = model.vision_heads
            head_dim = vision_width // num_heads
            q_w, k_w, v_w = jnp.split(src_value, 3, axis=0)
            w_map = {"query": q_w, "key": k_w, "value": v_w}
            src_value = jnp.transpose(w_map[flax_dst_key_tuple[-2]], (1, 0)).reshape(vision_width, num_heads, head_dim)
        elif hf_src_key_tuple[-1] == "in_proj_bias":
            num_heads = model.vision_heads
            head_dim = vision_width // num_heads
            q_b, k_b, v_b = jnp.split(src_value, 3, axis=0)
            b_map = {"query": q_b, "key": k_b, "value": v_b}
            src_value = b_map[flax_dst_key_tuple[-2]].reshape(num_heads, head_dim)
        elif hf_src_key_tuple[-1] == "weight" and src_value.ndim == 2:
            if "position_embedding" not in hf_src_key_as_string and "token_embedding" not in hf_src_key_as_string:
                src_value = jnp.transpose(src_value, (1, 0))
        if src_value.shape != dst_value_obj.value.shape:
            raise ValueError(f"Shape mismatch for {flax_dst_key_tuple} (Flax) vs {hf_src_key_as_string} (HF): {dst_value_obj.value.shape} (expected) != {src_value.shape} (actual)")

        src_value = src_value.astype(param_dtype)
        dst_value_obj.value = src_value

    nnx.update(model, nnx.from_flat_state(flax_model_params_fstate))

    hf_checkpoint_keys: Set[str] = set(params_fstate.keys())
    leftover_hf_keys = hf_checkpoint_keys - used_hf_keys
    known_unused_hf_buffer_keys = {
        "text_model.embeddings.position_ids",
        "vision_model.embeddings.position_ids",
    }
    unexpected_leftover_hf_keys = leftover_hf_keys - known_unused_hf_buffer_keys

    assert len(unexpected_leftover_hf_keys) == 0, f"Some unexpected HuggingFace checkpoint parameters were not used: {sorted(list(unexpected_leftover_hf_keys))}"
    model._original_config = config
    return model
