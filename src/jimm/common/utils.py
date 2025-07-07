import json
import os
from typing import Any, Dict, Tuple

import jax.numpy as jnp
from flax import nnx
from huggingface_hub import hf_hub_download
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array
from safetensors.flax import load_file as load_safetensors_flax_file


def sharded_init(init: nnx.Initializer, spec: P, mesh: Mesh | None) -> nnx.Initializer:
    """Create a sharded initializer if mesh is provided, otherwise return the original initializer.

    Args:
        init (nnx.Initializer): The initializer to shard.
        spec (P): The sharding specification.
        mesh (Mesh|None): The mesh to shard the initializer on. Defaults to None.

    Returns:
        nnx.Initializer: The possibly sharded initializer.
    """
    return nnx.with_partitioning(init, NamedSharding(mesh, spec)) if mesh is not None else init


def load_params_and_config(
    model_name_or_path: str,
    use_pytorch: bool = False,
    default_config_filename: str = "config.json",
    default_pytorch_filename: str = "pytorch_model.bin",
    default_safetensors_filename: str = "model.safetensors",
) -> Tuple[Dict[str, Array], Dict[str, Any]]:
    """Loads model parameters and configuration from local files or HuggingFace Hub.

    Args:
        model_name_or_path (str): Path to local weights/config or HuggingFace model ID.
        use_pytorch (bool): Whether to load from PyTorch weights. Defaults to False.
        default_config_filename (str): Default filename for config if model_name_or_path is a repo ID or local directory.
        default_pytorch_filename (str): Default filename for PyTorch weights if model_name_or_path is a repo ID or local directory.
        default_safetensors_filename (str): Default filename for safetensors if model_name_or_path is a repo ID.

    Returns:
        Tuple[Dict[str, Array], Dict[str, Any]]:
            A tuple containing the loaded parameters (params_fstate) and the configuration dictionary.
            Config is an empty dict ({}) if it could not be loaded by this utility.
    """
    params_fstate: Dict[str, Array] | None = None
    config: Dict[str, Any] = {}

    config_file_path: str | None = None
    weights_file_path: str | None = None

    if use_pytorch:
        import torch

        if os.path.isdir(model_name_or_path):
            config_file_path = os.path.join(model_name_or_path, default_config_filename)
            weights_file_path = os.path.join(model_name_or_path, default_pytorch_filename)
        else:
            config_file_path = hf_hub_download(repo_id=model_name_or_path, filename=default_config_filename)
            weights_file_path = hf_hub_download(repo_id=model_name_or_path, filename=default_pytorch_filename)

        if config_file_path and os.path.exists(config_file_path):
            with open(config_file_path, "r") as f:
                config = json.load(f)

        if weights_file_path and os.path.exists(weights_file_path):
            state_dict = torch.load(weights_file_path, map_location="cpu")
            params_fstate = {k: jnp.array(v.numpy()) for k, v in state_dict.items()}

    else:  # SafeTensors
        if os.path.exists(model_name_or_path) and os.path.isfile(model_name_or_path):  # Local safetensors file
            weights_file_path = model_name_or_path

            config_path_attempt1 = os.path.join(os.path.dirname(model_name_or_path), default_config_filename)
            if os.path.exists(config_path_attempt1):
                config_file_path = config_path_attempt1
            else:
                current_dir_name = os.path.basename(os.path.dirname(model_name_or_path))
                if current_dir_name == "model":
                    parent_dir_of_model_dir = os.path.dirname(os.path.dirname(model_name_or_path))
                    config_path_attempt2 = os.path.join(parent_dir_of_model_dir, default_config_filename)
                    if os.path.exists(config_path_attempt2):
                        config_file_path = config_path_attempt2

            if config_file_path and os.path.exists(config_file_path):
                with open(config_file_path, "r") as f:
                    config = json.load(f)

        else:  # HuggingFace Hub
            try:
                config_file_path = hf_hub_download(repo_id=model_name_or_path, filename=default_config_filename)
                with open(config_file_path, "r") as f:
                    config = json.load(f)
            except Exception:
                config = {}
            weights_file_path = hf_hub_download(repo_id=model_name_or_path, filename=default_safetensors_filename)

        if weights_file_path and os.path.exists(weights_file_path):
            params_fstate = load_safetensors_flax_file(weights_file_path)

    if params_fstate is None:
        raise ValueError(f"Could not load parameters from {model_name_or_path} (use_pytorch={use_pytorch})")

    return params_fstate, config
