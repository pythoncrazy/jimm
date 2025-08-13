import jax
import jax.numpy as jnp
from flax import nnx
from jimm.models import CLIP
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from safetensors.flax import save_file
from typing import Dict
from jaxtyping import Array

HF_MODEL_NAME = "openai/clip-vit-large-patch14"
USE_PYTORCH = True

devices = mesh_utils.create_device_mesh((1, jax.device_count()))
mesh = Mesh(devices, ("batch", "model"))


@nnx.jit
def create_sharded_model() -> CLIP:
    """Create and shard the CLIP model and optimizer following FSDP pattern.

    Returns:
        Tuple[CLIP, nnx.Optimizer]: Sharded model and optimizer
    """
    model = CLIP.from_pretrained(HF_MODEL_NAME, use_gradient_checkpointing=True, rngs=nnx.Rngs(0))
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state)
    return model


with mesh:
    model = create_sharded_model()


def convert_key_to_hf_format(key: str) -> str:
    """Convert JIMM parameter key to HuggingFace format.

    Args:
        key: JIMM parameter key

    Returns:
        HuggingFace format key
    """
    key = key.replace("text_model.layers.", "text_model.encoder.layers.")
    key = key.replace(".attn.query.", ".self_attn.q_proj.")
    key = key.replace(".attn.key.", ".self_attn.k_proj.")
    key = key.replace(".attn.value.", ".self_attn.v_proj.")
    key = key.replace(".attn.out.", ".self_attn.out_proj.")
    key = key.replace(".mlp.layers.0.", ".mlp.fc1.")
    key = key.replace(".mlp.layers.3.", ".mlp.fc2.")
    key = key.replace(".norm1.", ".layer_norm1.")
    key = key.replace(".norm2.", ".layer_norm2.")
    key = key.replace(".scale", ".weight")
    key = key.replace(".kernel", ".weight")

    special_mappings = {
        "ln_final.weight": "text_model.final_layer_norm.weight",
        "ln_final.bias": "text_model.final_layer_norm.bias",
        "vision_model.ln_pre.weight": "vision_model.pre_layrnorm.weight",
        "vision_model.ln_pre.bias": "vision_model.pre_layrnorm.bias",
        "vision_model.ln_post.weight": "vision_model.post_layernorm.weight",
        "vision_model.ln_post.bias": "vision_model.post_layernorm.bias",
        "vision_model.cls_token": "vision_model.embeddings.class_embedding",
        "vision_model.position_embeddings": "vision_model.embeddings.position_embedding.weight",
        "vision_model.patch_embeddings.weight": "vision_model.embeddings.patch_embedding.weight",
        "positional_embedding": "text_model.embeddings.position_embedding.weight",
        "text_position_ids": "text_model.embeddings.position_ids",
        "vision_model.vision_position_ids": "vision_model.embeddings.position_ids",
        "token_embedding.embedding": "text_model.embeddings.token_embedding.weight",
        "text_projection.weight": "text_projection.weight",
        "visual_projection.weight": "visual_projection.weight",
    }
    return special_mappings.get(key, key)


def convert_tensor_to_hf_format(hf_key: str, tensor: Array) -> Array:
    """Convert JIMM tensor to HuggingFace format.

    Args:
        hf_key: HuggingFace parameter key
        tensor: JIMM tensor

    Returns:
        HuggingFace format tensor
    """
    if ".self_attn.q_proj.weight" in hf_key or ".self_attn.k_proj.weight" in hf_key or ".self_attn.v_proj.weight" in hf_key:
        if tensor.ndim == 3:
            return tensor.reshape(-1, tensor.shape[0]).T
    elif ".self_attn.q_proj.bias" in hf_key or ".self_attn.k_proj.bias" in hf_key or ".self_attn.v_proj.bias" in hf_key:
        if tensor.ndim == 2:
            return tensor.flatten()
    elif ".self_attn.out_proj.weight" in hf_key:
        if tensor.ndim == 3:
            return tensor.reshape(tensor.shape[2], -1).T
    elif "patch_embedding.weight" in hf_key:
        if tensor.ndim == 4:
            return jnp.transpose(tensor, (3, 2, 0, 1))
    elif "class_embedding" in hf_key:
        if tensor.ndim == 3:
            return tensor.squeeze()
    elif "position_embedding.weight" in hf_key and "vision_model" in hf_key:
        if tensor.ndim == 3:
            return tensor.squeeze(0)
    elif "token_embedding.weight" in hf_key or "position_embedding.weight" in hf_key:
        return tensor
    elif hf_key.endswith(".weight") and tensor.ndim == 2:
        return tensor.T
    return tensor


def filter_tensors(state_dict: Dict) -> Dict[str, Array]:
    """Filter valid tensors from model state.

    Args:
        state_dict: Model state dictionary

    Returns:
        Filtered tensor dictionary
    """
    filtered = {}

    def process_item(key_name: str, value, prefix: str = ""):
        full_key = f"{prefix}.{key_name}" if prefix else key_name
        if "attn_mask" in full_key or "rngs" in full_key:
            return
        if isinstance(value, jax.Array):
            if "key<" not in str(value.dtype) and "prng" not in str(value.dtype).lower():
                filtered[full_key] = jax.device_get(value)
        elif isinstance(value, dict):
            for nested_key, nested_value in value.items():
                process_item(nested_key, nested_value, full_key)

    for key, value in state_dict.items():
        process_item(key, value)
    return filtered


def convert_state_to_hf_format(state_dict: Dict) -> Dict[str, Array]:
    """Convert JIMM model state to HuggingFace format.

    Args:
        state_dict: JIMM model state dictionary

    Returns:
        HuggingFace format state dictionary
    """
    tensor_state = filter_tensors(state_dict)
    hf_state = {}

    for jimm_key, tensor in tensor_state.items():
        hf_key = convert_key_to_hf_format(jimm_key)
        hf_tensor = convert_tensor_to_hf_format(hf_key, tensor)
        hf_state[hf_key] = hf_tensor

    return hf_state


graphdef, state = nnx.split(model)
pure_state = nnx.to_pure_dict(state)
hf_tensor_state = convert_state_to_hf_format(pure_state)

print(f"Filtered state contains {len(hf_tensor_state)} tensors")
print("Tensor keys preview:", list(hf_tensor_state.keys())[:5])
save_file(hf_tensor_state, "tmp/clip_model.safetensors")
