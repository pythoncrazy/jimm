import json
import os
from typing import Dict, Optional

import jax
import jax.numpy as jnp
from flax import nnx
from huggingface_hub import hf_hub_download
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.typing import DTypeLike
from jaxtyping import Array, Float
from safetensors.flax import load_file

from jimm.common.transformer import Transformer
from jimm.common.utils import sharded_init


class VisionTransformer(nnx.Module):
    """Vision Transformer (ViT) model for image classification.

    This implements the Vision Transformer as described in the paper
    "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
    """

    def __init__(
        self,
        num_classes: int = 1000,
        in_channels: int = 3,
        img_size: int = 224,
        patch_size: int = 16,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        hidden_size: int = 768,
        dropout_rate: float = 0.1,
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(0),
        mesh: Optional[Mesh] = None,
    ) -> None:
        """Initialize a Vision Transformer.

        Args:
            num_classes (int): Number of output classes
            in_channels (int): Number of input channels
            img_size (int): Size of the input image (assumed square)
            patch_size (int): Size of each patch (assumed square)
            num_layers (int): Number of transformer layers
            num_heads (int): Number of attention heads
            mlp_dim (int): Size of the MLP dimension
            hidden_size (int): Size of the hidden dimension
            dropout_rate (float): Dropout rate
            dtype (DTypeLike): Data type for computations
            param_dtype (DTypeLike): Data type for parameters
            rngs (nnx.Rngs): Random number generator keys
            mesh (Optional[Mesh]): Optional JAX device mesh for parameter sharding
        """
        n_patches = (img_size // patch_size) ** 2
        self.patch_embeddings = nnx.Conv(
            in_channels,
            hidden_size,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            kernel_init=sharded_init(nnx.initializers.xavier_uniform(), P(None, None, None, "model"), mesh),
            bias_init=sharded_init(nnx.initializers.zeros_init(), P("model"), mesh),
        )
        initializer = jax.nn.initializers.truncated_normal(stddev=0.02)
        # n_patches_plus_1 corresponds to n_patches + 1 (for CLS token)
        pos_emb_value_unsharded: Float[Array, "one n_patches_plus_1 hidden_size_dim"] = initializer(rngs.params(), (1, n_patches + 1, hidden_size), dtype=dtype)
        if mesh is not None:
            pos_emb_value_sharded = jax.device_put(pos_emb_value_unsharded, NamedSharding(mesh, P(None, None, "model")))
            self.position_embeddings = nnx.Param(pos_emb_value_sharded)
        else:
            self.position_embeddings = nnx.Param(pos_emb_value_unsharded)

        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

        cls_token_value_unsharded: Float[Array, "one one hidden_size_dim"] = jnp.zeros((1, 1, hidden_size), dtype=dtype)
        if mesh is not None:
            cls_token_value_sharded = jax.device_put(cls_token_value_unsharded, NamedSharding(mesh, P(None, None, "model")))
            self.cls_token = nnx.Param(cls_token_value_sharded)
        else:
            self.cls_token = nnx.Param(cls_token_value_unsharded)

        self.encoder = Transformer(
            width=hidden_size,
            mlp_dim=mlp_dim,
            layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            mesh=mesh,
        )

        self.final_norm = nnx.LayerNorm(
            hidden_size,
            epsilon=1e-12,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            scale_init=sharded_init(nnx.initializers.ones_init(), P("model"), mesh),
            bias_init=sharded_init(nnx.initializers.zeros_init(), P("model"), mesh),
        )
        self.classifier = nnx.Linear(
            hidden_size,
            num_classes,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            kernel_init=sharded_init(nnx.initializers.xavier_uniform(), P(None, "model"), mesh),
            bias_init=sharded_init(nnx.initializers.zeros_init(), P("model"), mesh),
        )

    def __call__(self, x: Float[Array, "batch height width channels"]) -> Float[Array, "batch num_classes"]:
        """Forward pass of the Vision Transformer.

        Args:
            x (Float[Array, "batch height width channels"]): Input tensor with shape [batch, height, width, channels]

        Returns:
            Float[Array, "batch num_classes"]: Output logits with shape [batch, num_classes]
        """
        patches = self.patch_embeddings(x)
        batch_size = patches.shape[0]
        patches = patches.reshape(batch_size, -1, patches.shape[-1])
        cls_token = jnp.tile(self.cls_token.value, [batch_size, 1, 1])
        x = jnp.concat([cls_token, patches], axis=1)
        embeddings = x + self.position_embeddings.value
        embeddings = self.dropout(embeddings)
        x = self.encoder(embeddings)
        x = self.final_norm(x)
        x = x[:, 0]
        return self.classifier(x)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, use_pytorch: bool = False, mesh: Optional[Mesh] = None, dtype: DTypeLike = jnp.float32) -> "VisionTransformer":
        """Load a pretrained Vision Transformer from a local path or HuggingFace Hub.

        Args:
            model_name_or_path (str): Path to local weights or HuggingFace model ID
            use_pytorch (bool): Whether to load from PyTorch weights
            mesh (Optional[Mesh]): Optional device mesh for parameter sharding
            dtype (DTypeLike): Data type for computations
        Returns:
            VisionTransformer: Initialized Vision Transformer with pretrained weights
        """
        params_fstate: Optional[Dict[str, Array]] = None
        hidden_size, num_classes, num_layers, num_heads, mlp_dim, patch_size, img_size = [None] * 7

        if use_pytorch:
            import torch

            if os.path.isdir(model_name_or_path):
                config_file_path = os.path.join(model_name_or_path, "config.json")
                weights_file_path = os.path.join(model_name_or_path, "pytorch_model.bin")
            else:
                repo_id = model_name_or_path
                config_file_path = hf_hub_download(repo_id=repo_id, filename="config.json")
                weights_file_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")

            with open(config_file_path, "r") as f:
                config = json.load(f)

            state_dict = torch.load(weights_file_path, map_location="cpu")
            params_fstate = {k: jnp.array(v.numpy()) for k, v in state_dict.items()}

            hidden_size = config["hidden_size"]
            num_classes = len(config["id2label"])
            num_layers = config["num_hidden_layers"]
            num_heads = config["num_attention_heads"]
            mlp_dim = config["intermediate_size"]
            patch_size = config["patch_size"]
            img_size = config["image_size"]

        elif os.path.exists(model_name_or_path) and os.path.isfile(model_name_or_path):
            safetensors_file_to_load = model_name_or_path
            params_fstate = load_file(safetensors_file_to_load)

            hidden_size = params_fstate["vit.embeddings.cls_token"].shape[-1]
            num_classes = params_fstate["classifier.bias"].shape[0]

            max_layer_idx = -1
            for k in params_fstate:
                if k.startswith("vit.encoder.layer."):
                    max_layer_idx = max(max_layer_idx, int(k.split(".")[3]))
            num_layers = max_layer_idx + 1

            mlp_dim = params_fstate["vit.encoder.layer.0.intermediate.dense.weight"].shape[0]

            assumed_head_dim = 64
            num_heads = hidden_size // assumed_head_dim

            patch_kernel_shape = params_fstate["vit.embeddings.patch_embeddings.projection.weight"].shape
            patch_size = patch_kernel_shape[2]

            num_patches_from_embeddings = params_fstate["vit.embeddings.position_embeddings"].shape[1] - 1
            img_size_dim = int(jnp.sqrt(num_patches_from_embeddings))
            img_size = img_size_dim * patch_size
        else:
            repo_id = model_name_or_path
            config_file_path = hf_hub_download(repo_id=repo_id, filename="config.json")
            safetensors_file_to_load = hf_hub_download(repo_id=repo_id, filename="model.safetensors")

            with open(config_file_path, "r") as f:
                config = json.load(f)

            params_fstate = load_file(safetensors_file_to_load)

            hidden_size = config["hidden_size"]
            num_classes = len(config["id2label"])
            num_layers = config["num_hidden_layers"]
            num_heads = config["num_attention_heads"]
            mlp_dim = config["intermediate_size"]
            patch_size = config["patch_size"]
            img_size = config["image_size"]

        model = cls(
            num_classes=num_classes,
            img_size=img_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            hidden_size=hidden_size,
            mesh=mesh,
            dtype=dtype,
            param_dtype=dtype,
        )

        flax_model_params_fstate = dict(nnx.to_flat_state(nnx.state(model, nnx.Param)))

        def hf_param_name(name: str) -> str:
            """Converts a Flax parameter name component to its HuggingFace equivalent.

            Specifically, "kernel" and "scale" are mapped to "weight". Other names
            are returned unchanged.

            Args:
                name (str): The Flax parameter name component (e.g., "kernel", "scale", "bias").

            Returns:
                str: The corresponding HuggingFace parameter name component (e.g., "weight", "bias").
            """
            return "weight" if name in ["kernel", "scale"] else name

        hidden_size_per_head = hidden_size // num_heads

        mapping_list = [
            (("cls_token",), ("vit", "embeddings", "cls_token")),
            (("position_embeddings",), ("vit", "embeddings", "position_embeddings")),
        ]
        mapping_list.extend(
            [
                (("patch_embeddings", "kernel"), ("vit", "embeddings", "patch_embeddings", "projection", "weight")),  # type: ignore
                (("patch_embeddings", "bias"), ("vit", "embeddings", "patch_embeddings", "projection", "bias")),  # type: ignore
            ]
        )
        mapping_list.extend([(("classifier", "kernel"), ("classifier", "weight")), (("classifier", "bias"), ("classifier", "bias"))])  # type: ignore
        mapping_list.extend([(("final_norm", "scale"), ("vit", "layernorm", "weight")), (("final_norm", "bias"), ("vit", "layernorm", "bias"))])  # type: ignore

        for i in range(num_layers):
            flax_base = ("encoder", "layers", i)
            hf_base = ("vit", "encoder", "layer", str(i))
            mapping_list.extend(
                [(flax_base + ("attn", y_type, p_name), hf_base + ("attention", "attention", y_type, hf_param_name(p_name))) for p_name in ["kernel", "bias"] for y_type in ["key", "value", "query"]]  # type: ignore
            )
            mapping_list.extend([(flax_base + ("attn", "out", p_name), hf_base + ("attention", "output", "dense", hf_param_name(p_name))) for p_name in ["kernel", "bias"]])  # type: ignore
            mapping_list.extend(
                [
                    (flax_base + ("mlp", "layers", y1_idx, p_name), hf_base + (y2_name, "dense", hf_param_name(p_name)))  # type: ignore
                    for p_name in ["kernel", "bias"]
                    for y1_idx, y2_name in [(0, "intermediate"), (3, "output")]  # type: ignore
                ]
            )  # type: ignore
            mapping_list.extend(
                [
                    (flax_base + (norm_flax, p_name), hf_base + (norm_hf, hf_param_name(p_name)))
                    for p_name in ["scale", "bias"]
                    for norm_flax, norm_hf in [("norm1", "layernorm_before"), ("norm2", "layernorm_after")]  # type: ignore
                ]  # type: ignore
            )  # type: ignore
        params_name_mapping = dict(mapping_list)
        nonvisited = set(flax_model_params_fstate.keys())

        for flax_dst_key_tuple, hf_src_key_tuple in params_name_mapping.items():
            assert flax_dst_key_tuple in flax_model_params_fstate, flax_dst_key_tuple
            hf_src_key_as_string = ".".join(hf_src_key_tuple)
            assert hf_src_key_as_string in params_fstate, f"HF key '{hf_src_key_as_string}' (from Flax key {flax_dst_key_tuple}) not found in loaded safetensors."
            nonvisited.remove(flax_dst_key_tuple)
            src_value: Array = params_fstate[hf_src_key_as_string]

            dst_value_obj = flax_model_params_fstate[flax_dst_key_tuple]
            original_param_sharding = dst_value_obj.value.sharding

            if flax_dst_key_tuple == ("patch_embeddings", "kernel"):
                src_value = jnp.transpose(src_value, (2, 3, 1, 0))
            elif hf_src_key_tuple[-1] == "weight" and hf_src_key_tuple[-2] in ("key", "value", "query"):
                src_value = jnp.transpose(src_value, (1, 0))
                src_value = src_value.reshape((hidden_size, num_heads, hidden_size_per_head))
            elif hf_src_key_tuple[-1] == "bias" and hf_src_key_tuple[-2] in ("key", "value", "query"):
                src_value = src_value.reshape((num_heads, hidden_size_per_head))
            elif hf_src_key_tuple[-4:] == ("attention", "output", "dense", "weight"):
                src_value = jnp.transpose(src_value, (1, 0))
                src_value = src_value.reshape((num_heads, hidden_size_per_head, hidden_size))
            elif hf_src_key_tuple[-1] == "weight" and src_value.ndim == 2:
                src_value = jnp.transpose(src_value, (1, 0))

            assert src_value.shape == dst_value_obj.value.shape, f"Shape mismatch for {flax_dst_key_tuple} (Flax) vs {hf_src_key_as_string} (HF): {dst_value_obj.value.shape} != {src_value.shape}"

            sharded_new_value: Array = jax.device_put(src_value, original_param_sharding)
            dst_value_obj.value = sharded_new_value

            assert jnp.allclose(dst_value_obj.value.mean(), src_value.mean()), (dst_value_obj.value.mean(), src_value.mean())

        assert len(nonvisited) == 0, f"Some Flax model parameters were not visited: {nonvisited}"
        nnx.update(model, nnx.from_flat_state(flax_model_params_fstate))

        del flax_model_params_fstate
        del params_fstate
        if "config" in locals():
            del config
        if "state_dict" in locals():
            del state_dict
        if "torch" in locals() and hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return model
