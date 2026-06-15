import json
import os
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib
from jax.typing import DTypeLike
from jaxtyping import Array
from safetensors.flax import save_file as save_safetensors

from jimm.common.loading_utils import apply_mapping, expand_scanned_layers, load_params_and_config
from jimm.common.sharding import ShardingSpec
from jimm.common.utils import convert_key_to_hf_format, filter_tensors
from jimm.models.dinov3.sharding import DINOv3Sharding

if TYPE_CHECKING:
    from jimm.models.dinov3.dinov3_model import DINOv3Model


class _Transform(Enum):
    LINEAR = ((1, 0), None, False)
    CONV2D = ((2, 3, 1, 0), None, False)
    DEFAULT = (None, None, False)


def _get_key_and_transform_mapping(use_gated_mlp: bool) -> dict[str, tuple[str, Any]]:
    """Return regex-based key mapping from HuggingFace to Flax format for DINOv3.

    Args:
        use_gated_mlp (bool): Whether the model uses gated MLP (affects MLP key names).

    Returns:
        dict[str, tuple[str, Any]]: Dict of {regex_pattern: (flax_key_template, _Transform)}.
    """

    mapping = {
        r"embeddings\.cls_token$": ("encoder.cls_token", _Transform.DEFAULT),
        r"embeddings\.register_tokens$": ("encoder.register_tokens", _Transform.DEFAULT),
        r"embeddings\.patch_embeddings\.weight$": ("encoder.patch_embeddings.kernel", _Transform.CONV2D),
        r"embeddings\.patch_embeddings\.bias$": ("encoder.patch_embeddings.bias", _Transform.DEFAULT),
        r"norm\.weight$": ("encoder.ln_post.scale", _Transform.DEFAULT),
        r"norm\.bias$": ("encoder.ln_post.bias", _Transform.DEFAULT),
        r"layer\.([0-9]+)\.attention\.q_proj\.weight$": (r"encoder.layers_\1.attn.query.kernel", _Transform.LINEAR),
        r"layer\.([0-9]+)\.attention\.q_proj\.bias$": (r"encoder.layers_\1.attn.query.bias", _Transform.DEFAULT),
        r"layer\.([0-9]+)\.attention\.k_proj\.weight$": (r"encoder.layers_\1.attn.key.kernel", _Transform.LINEAR),
        r"layer\.([0-9]+)\.attention\.v_proj\.weight$": (r"encoder.layers_\1.attn.value.kernel", _Transform.LINEAR),
        r"layer\.([0-9]+)\.attention\.v_proj\.bias$": (r"encoder.layers_\1.attn.value.bias", _Transform.DEFAULT),
        r"layer\.([0-9]+)\.attention\.o_proj\.weight$": (r"encoder.layers_\1.attn.out.kernel", _Transform.LINEAR),
        r"layer\.([0-9]+)\.attention\.o_proj\.bias$": (r"encoder.layers_\1.attn.out.bias", _Transform.DEFAULT),
        r"layer\.([0-9]+)\.layer_scale1\.lambda1$": (r"encoder.layers_\1.layer_scale1", _Transform.DEFAULT),
        r"layer\.([0-9]+)\.layer_scale2\.lambda1$": (r"encoder.layers_\1.layer_scale2", _Transform.DEFAULT),
        r"layer\.([0-9]+)\.norm1\.weight$": (r"encoder.layers_\1.norm1.scale", _Transform.DEFAULT),
        r"layer\.([0-9]+)\.norm1\.bias$": (r"encoder.layers_\1.norm1.bias", _Transform.DEFAULT),
        r"layer\.([0-9]+)\.norm2\.weight$": (r"encoder.layers_\1.norm2.scale", _Transform.DEFAULT),
        r"layer\.([0-9]+)\.norm2\.bias$": (r"encoder.layers_\1.norm2.bias", _Transform.DEFAULT),
    }

    if use_gated_mlp:
        mapping.update(
            {
                r"layer\.([0-9]+)\.mlp\.gate_proj\.weight$": (r"encoder.layers_\1.gate.kernel", _Transform.LINEAR),
                r"layer\.([0-9]+)\.mlp\.gate_proj\.bias$": (r"encoder.layers_\1.gate.bias", _Transform.DEFAULT),
                r"layer\.([0-9]+)\.mlp\.up_proj\.weight$": (r"encoder.layers_\1.up.kernel", _Transform.LINEAR),
                r"layer\.([0-9]+)\.mlp\.up_proj\.bias$": (r"encoder.layers_\1.up.bias", _Transform.DEFAULT),
                r"layer\.([0-9]+)\.mlp\.down_proj\.weight$": (r"encoder.layers_\1.down.kernel", _Transform.LINEAR),
                r"layer\.([0-9]+)\.mlp\.down_proj\.bias$": (r"encoder.layers_\1.down.bias", _Transform.DEFAULT),
            }
        )
    else:
        mapping.update(
            {
                r"layer\.([0-9]+)\.mlp\.up_proj\.weight$": (r"encoder.layers_\1.mlp.layers.0.kernel", _Transform.LINEAR),
                r"layer\.([0-9]+)\.mlp\.up_proj\.bias$": (r"encoder.layers_\1.mlp.layers.0.bias", _Transform.DEFAULT),
                r"layer\.([0-9]+)\.mlp\.down_proj\.weight$": (r"encoder.layers_\1.mlp.layers.3.kernel", _Transform.LINEAR),
                r"layer\.([0-9]+)\.mlp\.down_proj\.bias$": (r"encoder.layers_\1.mlp.layers.3.bias", _Transform.DEFAULT),
            }
        )

    return mapping


def _create_dinov3_config(model: "DINOv3Model") -> dict[str, Any]:
    """Create HuggingFace-compatible config dictionary for DINOv3.

    Args:
        model (DINOv3Model): The DINOv3Model instance.

    Returns:
        dict[str, Any]: Configuration dictionary in HuggingFace DINOv3 format.
    """
    if model._original_config is not None:
        return model._original_config.copy()

    enc = model.encoder
    patch_size = enc.patch_embeddings.kernel_size[0]
    hidden_size = enc.patch_embeddings.out_features
    use_gated_mlp = enc.use_gated_mlp

    return {
        "model_type": "dinov3_vit",
        "hidden_size": hidden_size,
        "num_hidden_layers": enc.num_layers,
        "num_attention_heads": enc.num_heads,
        "intermediate_size": enc.mlp_dim,
        "hidden_act": enc.hidden_act,
        "layer_norm_eps": enc.layernorm_epsilon,
        "rope_theta": enc.rope_theta,
        "image_size": enc.img_size,
        "patch_size": patch_size,
        "num_channels": enc.patch_embeddings.in_features,
        "layerscale_value": model._layer_scale_init,
        "use_gated_mlp": use_gated_mlp,
        "num_register_tokens": enc.num_register_tokens,
        "use_patch_bias": enc.patch_embeddings.use_bias,
        "key_bias": False,
        "initializer_range": 0.02,
    }


def _convert_dinov3_tensor_to_hf_format(hf_key: str, tensor: Array) -> Array:
    """Convert DINOv3 tensor from Flax format to HuggingFace format.

    Args:
        hf_key (str): The HuggingFace key for the tensor.
        tensor (Array): The tensor to convert.

    Returns:
        Array: The converted tensor in HuggingFace format.
    """
    if "patch_embeddings.weight" in hf_key and tensor.ndim == 4:
        return jnp.transpose(tensor, (3, 2, 0, 1))
    if any(s in hf_key for s in (".q_proj.weight", ".k_proj.weight", ".v_proj.weight")):
        if tensor.ndim == 3:
            return jnp.transpose(tensor.reshape(tensor.shape[0], -1), (1, 0))
    elif any(s in hf_key for s in (".q_proj.bias", ".v_proj.bias")):
        if tensor.ndim == 2:
            return tensor.flatten()
    elif ".o_proj.weight" in hf_key:
        if tensor.ndim == 3:
            return jnp.transpose(tensor.reshape(-1, tensor.shape[2]), (1, 0))
    elif hf_key.endswith(".weight") and tensor.ndim == 2:
        return jnp.transpose(tensor, (1, 0))
    return tensor


def save_pretrained(model: "DINOv3Model", save_directory: str) -> None:
    """Save the DINOv3 model weights and config in HuggingFace format.

    Args:
        model (DINOv3Model): The DINOv3Model instance to save.
        save_directory (str): Directory path where the model will be saved.
    """
    enc = model.encoder
    use_gated_mlp = enc.use_gated_mlp

    _SPECIAL_MAPPINGS = {
        "encoder.cls_token": "embeddings.cls_token",
        "encoder.register_tokens": "embeddings.register_tokens",
        "encoder.patch_embeddings.weight": "embeddings.patch_embeddings.weight",
        "encoder.patch_embeddings.bias": "embeddings.patch_embeddings.bias",
        "encoder.ln_post.weight": "norm.weight",
        "encoder.ln_post.bias": "norm.bias",
    }
    _SPECIAL_RENAMINGS: dict[str, str] = {
        ".attn.query.": ".attention.q_proj.",
        ".attn.key.": ".attention.k_proj.",
        ".attn.value.": ".attention.v_proj.",
        ".attn.out.": ".attention.o_proj.",
        ".layer_scale1": ".layer_scale1.lambda1",
        ".layer_scale2": ".layer_scale2.lambda1",
    }
    if use_gated_mlp:
        _SPECIAL_RENAMINGS.update(
            {
                ".gate.": ".mlp.gate_proj.",
                ".up.": ".mlp.up_proj.",
                ".down.": ".mlp.down_proj.",
            }
        )
    else:
        _SPECIAL_RENAMINGS.update(
            {
                ".mlp.layers.0.": ".mlp.up_proj.",
                ".mlp.layers.3.": ".mlp.down_proj.",
            }
        )
    for i in range(enc.num_layers):
        _SPECIAL_RENAMINGS[f"encoder.layers_{i}."] = f"layer.{i}."

    os.makedirs(save_directory, exist_ok=True)

    config = _create_dinov3_config(model)
    with open(os.path.join(save_directory, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    _, state = nnx.split(model)
    state_dict = nnx.to_pure_dict(state)
    state_dict["encoder"] = expand_scanned_layers(state_dict["encoder"])

    tensor_state = filter_tensors(state_dict)
    hf_state: dict[str, Array] = {}
    for jimm_key, tensor in tensor_state.items():
        if ".attn.key.bias" in jimm_key:
            continue
        hf_key = convert_key_to_hf_format(jimm_key, _SPECIAL_MAPPINGS, _SPECIAL_RENAMINGS)
        hf_state[hf_key] = _convert_dinov3_tensor_to_hf_format(hf_key, tensor)

    save_safetensors(hf_state, os.path.join(save_directory, "model.safetensors"))


def load_from_pretrained(
    cls,
    model_name_or_path: str,
    use_pytorch: bool = False,
    rngs: rnglib.Rngs | None = None,
    dtype: DTypeLike = jnp.float32,
    param_dtype: DTypeLike = jnp.float32,
    sharding: ShardingSpec = DINOv3Sharding(),
    use_gradient_checkpointing: bool = False,
    attention_fn: Callable[..., Any] | None = None,
) -> "DINOv3Model":
    """Load a pretrained DINOv3 model from a local path or HuggingFace Hub.

    Args:
        cls: The DINOv3Model class.
        model_name_or_path (str): Local directory or HuggingFace model ID.
        use_pytorch (bool, optional): Load from PyTorch weights. Defaults to False.
        rngs (rnglib.Rngs | None, optional): RNG state. Defaults to nnx.Rngs(0).
        dtype (DTypeLike, optional): Computation dtype. Defaults to jnp.float32.
        param_dtype (DTypeLike, optional): Parameter dtype. Defaults to jnp.float32.
        sharding (ShardingSpec, optional): Sharding specification. Defaults to DINOv3Sharding.
        use_gradient_checkpointing (bool, optional): Enable gradient checkpointing. Defaults to False.
        attention_fn (Callable[..., Any] | None, optional): Custom attention function. Defaults to None.

    Returns:
        DINOv3Model: Model with pretrained weights loaded.
    """
    if rngs is None:
        rngs = nnx.Rngs(0)

    params_fstate, config_dict = load_params_and_config(model_name_or_path, use_pytorch)

    parsed = cls._parse_config(config_dict)
    model = cls(
        **parsed,
        use_gradient_checkpointing=use_gradient_checkpointing,
        attention_fn=attention_fn,
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
        sharding=sharding,
    )

    apply_mapping(model, params_fstate, _get_key_and_transform_mapping(parsed["use_gated_mlp"]), param_dtype)
    model._original_config = config_dict
    model.eval()
    return model
