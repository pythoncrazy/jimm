import json
import os
from typing import TYPE_CHECKING, Any, Set

import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib
from jax.sharding import Mesh
from jax.typing import DTypeLike
from jaxtyping import Array
from safetensors.flax import save_file as save_safetensors

from jimm.common.utils import convert_key_to_hf_format, filter_tensors, load_params_and_config

if TYPE_CHECKING:
    from jimm.models import VisionTransformer


def _create_config(model: "VisionTransformer") -> dict[str, Any]:
    """Create HuggingFace-compatible config dictionary.

    Args:
        model (VisionTransformer): The VisionTransformer model instance.

    Returns:
        dict[str, Any]: Configuration dictionary in HuggingFace format.
    """
    if model._original_config is not None:
        return model._original_config.copy()

    return {
        "model_type": "vit",
        "architectures": ["ViTForImageClassification"],
        "hidden_size": model.encoder.hidden_size,
        "num_hidden_layers": len(model.encoder.transformer.blocks.layers),
        "num_attention_heads": model.encoder.encoder.layers[0].attn.num_heads,
        "intermediate_size": model.encoder.encoder.layers[0].mlp.layers[0].out_features,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "image_size": model.encoder.img_size,
        "patch_size": model.encoder.patch_size,
        "num_channels": model.encoder.patch_embeddings.in_features // (model.encoder.patch_size**2),
        "qkv_bias": True,
    }


def _convert_vit_tensor_to_hf_format(hf_key: str, tensor: Array, num_heads: int, hidden_size: int, head_dim: int) -> Array:
    """Convert ViT tensor from Flax format to HuggingFace format.

    This reverses the transformations done in from_pretrained method.

    Args:
        hf_key (str): The HuggingFace key for the tensor.
        tensor (Array): The tensor to convert.
        num_heads (int): Number of attention heads.
        hidden_size (int): Hidden dimension size.
        head_dim (int): Dimension of each attention head.

    Returns:
        Array: The converted tensor in HuggingFace format.
    """
    if ".attention.attention.query.weight" in hf_key or ".attention.attention.key.weight" in hf_key or ".attention.attention.value.weight" in hf_key:
        if tensor.ndim == 3 and tensor.shape == (hidden_size, num_heads, head_dim):
            tensor = tensor.reshape((hidden_size, hidden_size))
            tensor = jnp.transpose(tensor, (1, 0))
            return tensor

    elif ".attention.attention.query.bias" in hf_key or ".attention.attention.key.bias" in hf_key or ".attention.attention.value.bias" in hf_key:
        if tensor.ndim == 2 and tensor.shape == (num_heads, head_dim):
            return tensor.reshape((hidden_size,))

    elif ".attention.output.dense.weight" in hf_key:
        if tensor.ndim == 3 and tensor.shape == (num_heads, head_dim, hidden_size):
            tensor = tensor.reshape((hidden_size, hidden_size))
            tensor = jnp.transpose(tensor, (1, 0))
            return tensor

    elif "vit.embeddings.patch_embeddings.projection.weight" in hf_key:
        if tensor.ndim == 4:
            return jnp.transpose(tensor, (3, 2, 0, 1))

    elif hf_key.endswith(".weight") and tensor.ndim == 2:
        return jnp.transpose(tensor, (1, 0))

    return tensor


def save_pretrained(model: "VisionTransformer", save_directory: str):
    """Save the model weights and config in HuggingFace format.

    Args:
        model (VisionTransformer): The VisionTransformer model instance to save.
        save_directory (str): Directory path where the model will be saved.
    """
    _SPECIAL_MAPPINGS = {
        "encoder.cls_token": "vit.embeddings.cls_token",
        "encoder.position_embeddings": "vit.embeddings.position_embeddings",
        "encoder.patch_embeddings.weight": "vit.embeddings.patch_embeddings.projection.weight",
        "encoder.patch_embeddings.bias": "vit.embeddings.patch_embeddings.projection.bias",
        "classifier.weight": "classifier.weight",
        "classifier.bias": "classifier.bias",
        "encoder.ln_post.weight": "vit.layernorm.weight",
        "encoder.ln_post.bias": "vit.layernorm.bias",
    }

    _SPECIAL_RENAMINGS = {
        "encoder.encoder.layers": "vit.encoder.layer",
        ".attn.query.": ".attention.attention.query.",
        ".attn.key.": ".attention.attention.key.",
        ".attn.value.": ".attention.attention.value.",
        ".attn.out.": ".attention.output.dense.",
        ".mlp.layers.0.": ".intermediate.dense.",
        ".mlp.layers.3.": ".output.dense.",
        ".norm1.": ".layernorm_before.",
        ".norm2.": ".layernorm_after.",
    }
    # Fix layer numbering: layer_0 -> layer.0
    for i in range(100):
        _SPECIAL_RENAMINGS[f"layer_{i}."] = f"layer.{i}."

    os.makedirs(save_directory, exist_ok=True)

    config = _create_config(model)
    if jax.process_index() == 0:
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    _, state = nnx.split(model)
    state_dict = nnx.to_pure_dict(state)

    num_heads = model.encoder.encoder.layers_0.attn.num_heads
    hidden_size = model.encoder.encoder.width
    head_dim = hidden_size // num_heads

    tensor_state = filter_tensors(state_dict)
    hf_state = {}

    for jimm_key, tensor in tensor_state.items():
        hf_key = convert_key_to_hf_format(jimm_key, _SPECIAL_MAPPINGS, _SPECIAL_RENAMINGS)
        hf_tensor = _convert_vit_tensor_to_hf_format(hf_key, tensor, num_heads, hidden_size, head_dim)
        hf_state[hf_key] = hf_tensor

    if jax.process_index() == 0:
        save_safetensors(hf_state, os.path.join(save_directory, "model.safetensors"))


def load_from_pretrained(
    cls,
    model_name_or_path: str,
    use_pytorch: bool = False,
    mesh: Mesh | None = None,
    dtype: DTypeLike = jnp.float32,
    param_dtype: DTypeLike = jnp.float32,
    use_gradient_checkpointing: bool = False,
    rngs: rnglib.Rngs = nnx.Rngs(0),
) -> "VisionTransformer":
    """Load a pretrained Vision Transformer from a local path or HuggingFace Hub.

    Args:
        cls: The VisionTransformer class.
        model_name_or_path (str): Path to local weights or HuggingFace model ID.
        use_pytorch (bool): Whether to load from PyTorch weights. Defaults to False.
        mesh (Mesh | None): Optional device mesh for parameter sharding. Defaults to None.
        dtype (DTypeLike): Data type for computations. Defaults to jnp.float32.
        param_dtype (DTypeLike): Data type for parameters. Defaults to jnp.float32.
        use_gradient_checkpointing (bool): Whether to use gradient checkpointing. Defaults to False.
        rngs (rnglib.Rngs): Random number generator keys. Defaults to nnx.Rngs(0).

    Returns:
        VisionTransformer: Initialized Vision Transformer with pretrained weights
    """
    params_fstate, config_dict = load_params_and_config(model_name_or_path, use_pytorch)

    config: dict[str, Any] = config_dict

    hidden_size_val: int
    num_classes_val: int
    num_layers_val: int
    num_heads_val: int
    mlp_dim_val: int
    patch_size_val: int
    img_size_val: int
    use_quick_gelu_val: bool = False

    if config:
        hidden_size_val = config["hidden_size"]
        num_classes_val = len(config["id2label"]) if "id2label" in config else config.get("num_labels", 1000)
        num_layers_val = config["num_hidden_layers"]
        num_heads_val = config["num_attention_heads"]
        mlp_dim_val = config["intermediate_size"]
        patch_size_val = config["patch_size"]
        img_size_val = config["image_size"]
        if "hidden_act" in config and config["hidden_act"] == "quick_gelu":
            use_quick_gelu_val = True
        elif "hidden_act" in config and config["hidden_act"] != "gelu":
            print(f"Warning: Unexpected hidden_act '{config['hidden_act']}' in config, defaulting to standard GELU.")

    elif not use_pytorch and (os.path.exists(model_name_or_path) and os.path.isfile(model_name_or_path)):
        hidden_size_val = params_fstate["vit.embeddings.cls_token"].shape[-1]
        num_classes_val = params_fstate["classifier.bias"].shape[0]

        max_layer_idx = -1
        for k in params_fstate:
            if k.startswith("vit.encoder.layer."):
                max_layer_idx = max(max_layer_idx, int(k.split(".")[3]))
        num_layers_val = max_layer_idx + 1

        mlp_dim_val = params_fstate["vit.encoder.layer.0.intermediate.dense.weight"].shape[0]

        assumed_head_dim = 64
        num_heads_val = hidden_size_val // assumed_head_dim

        patch_kernel_shape = params_fstate["vit.embeddings.patch_embeddings.projection.weight"].shape
        patch_size_val = patch_kernel_shape[2]

        num_patches_from_embeddings = params_fstate["vit.embeddings.position_embeddings"].shape[1] - 1
        img_size_dim = int(jnp.sqrt(num_patches_from_embeddings))
        img_size_val = img_size_dim * patch_size_val
    else:
        raise ValueError(f"Could not load or infer configuration for {model_name_or_path}")

    if not all(v is not None for v in [hidden_size_val, num_classes_val, num_layers_val, num_heads_val, mlp_dim_val, patch_size_val, img_size_val]):
        raise ValueError(f"One or more configuration parameters could not be determined for {model_name_or_path}")

    model = cls(
        num_classes=num_classes_val,
        img_size=img_size_val,
        patch_size=patch_size_val,
        num_layers=num_layers_val,
        num_heads=num_heads_val,
        mlp_dim=mlp_dim_val,
        hidden_size=hidden_size_val,
        use_quick_gelu=use_quick_gelu_val,
        mesh=mesh,
        dtype=dtype,
        param_dtype=dtype,
        rngs=rngs,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )

    flax_model_params_fstate = dict(nnx.to_flat_state(nnx.state(model, nnx.Param)))

    def hf_param_name(name: str) -> str:
        return "weight" if name in ["kernel", "scale"] else name

    hidden_size_per_head = hidden_size_val // num_heads_val

    mapping_list = [
        (("encoder", "cls_token"), ("vit", "embeddings", "cls_token")),
        (("encoder", "position_embeddings"), ("vit", "embeddings", "position_embeddings")),
        (("encoder", "patch_embeddings", "kernel"), ("vit", "embeddings", "patch_embeddings", "projection", "weight")),
        (("encoder", "patch_embeddings", "bias"), ("vit", "embeddings", "patch_embeddings", "projection", "bias")),
        (("classifier", "kernel"), ("classifier", "weight")),
        (("classifier", "bias"), ("classifier", "bias")),
        (("encoder", "ln_post", "scale"), ("vit", "layernorm", "weight")),
        (("encoder", "ln_post", "bias"), ("vit", "layernorm", "bias")),
    ]

    for i in range(num_layers_val):
        flax_base = ("encoder", "encoder", f"layers_{i}")
        hf_base = ("vit", "encoder", "layer", str(i))
        mapping_list.extend(
            [(flax_base + ("attn", y_type, p_name), hf_base + ("attention", "attention", y_type, hf_param_name(p_name))) for p_name in ["kernel", "bias"] for y_type in ["key", "value", "query"]]
        )
        mapping_list.extend([(flax_base + ("attn", "out", p_name), hf_base + ("attention", "output", "dense", hf_param_name(p_name))) for p_name in ["kernel", "bias"]])
        mapping_list.extend(
            [
                (flax_base + ("mlp", "layers", y1_idx, p_name), hf_base + (y2_name, "dense", hf_param_name(p_name)))
                for p_name in ["kernel", "bias"]
                for y1_idx, y2_name in [(0, "intermediate"), (3, "output")]
            ]
        )
        mapping_list.extend(
            [
                (flax_base + (norm_flax, p_name), hf_base + (norm_hf, hf_param_name(p_name)))
                for p_name in ["scale", "bias"]
                for norm_flax, norm_hf in [("norm1", "layernorm_before"), ("norm2", "layernorm_after")]
            ]
        )
    params_name_mapping = dict(mapping_list)
    nonvisited = set(flax_model_params_fstate.keys())
    nonvisited.discard(("encoder", "vision_position_ids"))
    used_hf_keys: Set[str] = set()

    for flax_dst_key_tuple, hf_src_key_tuple in params_name_mapping.items():
        assert flax_dst_key_tuple in flax_model_params_fstate, flax_dst_key_tuple
        hf_src_key_as_string = ".".join(hf_src_key_tuple)
        used_hf_keys.add(hf_src_key_as_string)
        assert hf_src_key_as_string in params_fstate, f"HF key '{hf_src_key_as_string}' (from Flax key {flax_dst_key_tuple}) not found in loaded weights."
        nonvisited.remove(flax_dst_key_tuple)
        src_value: Array = params_fstate[hf_src_key_as_string]

        dst_value_obj = flax_model_params_fstate[flax_dst_key_tuple]

        if flax_dst_key_tuple == ("encoder", "patch_embeddings", "kernel"):
            src_value = jnp.transpose(src_value, (2, 3, 1, 0))
        elif hf_src_key_tuple[-1] == "weight" and hf_src_key_tuple[-2] in ("key", "value", "query"):
            src_value = jnp.transpose(src_value, (1, 0))
            src_value = src_value.reshape((hidden_size_val, num_heads_val, hidden_size_per_head))
        elif hf_src_key_tuple[-1] == "bias" and hf_src_key_tuple[-2] in ("key", "value", "query"):
            src_value = src_value.reshape((num_heads_val, hidden_size_per_head))
        elif hf_src_key_tuple[-4:] == ("attention", "output", "dense", "weight"):
            src_value = jnp.transpose(src_value, (1, 0))
            src_value = src_value.reshape((num_heads_val, hidden_size_per_head, hidden_size_val))
        elif hf_src_key_tuple[-1] == "weight" and src_value.ndim == 2:
            src_value = jnp.transpose(src_value, (1, 0))

        assert src_value.shape == dst_value_obj.value.shape, f"Shape mismatch for {flax_dst_key_tuple} (Flax) vs {hf_src_key_as_string} (HF): {dst_value_obj.value.shape} != {src_value.shape}"
        src_value = src_value.astype(param_dtype)
        dst_value_obj.value = src_value

    assert len(nonvisited) == 0, f"Some Flax model parameters were not visited: {nonvisited}"

    used_hf_keys.add("encoder.vision_position_ids")

    leftover_hf_keys = set(params_fstate.keys()) - used_hf_keys

    assert len(leftover_hf_keys) == 0, f"Some unexpected HuggingFace checkpoint parameters were not used: {sorted(list(leftover_hf_keys))}"
    nnx.update(model, nnx.from_flat_state(flax_model_params_fstate))

    model._original_config = config

    del flax_model_params_fstate
    del params_fstate
    return model
