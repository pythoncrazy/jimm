import os
from typing import Any, Set

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax.typing import DTypeLike
from jaxtyping import Array, Float

from jimm.common.utils import load_params_and_config, shard_model, sharded_init
from jimm.common.vit import VisionTransformerBase


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
        use_quick_gelu: bool = False,
        do_classification: bool = True,
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(0),
        mesh: Mesh | None = None,
    ) -> None:
        """Initialize a Vision Transformer.

        Args:
            num_classes (int): Number of output classes. Defaults to 1000.
            in_channels (int): Number of input channels. Defaults to 3.
            img_size (int): Size of the input image (assumed square). Defaults to 224.
            patch_size (int): Size of each patch (assumed square). Defaults to 16.
            num_layers (int): Number of transformer layers. Defaults to 12.
            num_heads (int): Number of attention heads. Defaults to 12.
            mlp_dim (int): Size of the MLP dimension. Defaults to 3072.
            hidden_size (int): Size of the hidden dimension. Defaults to 768.
            dropout_rate (float): Dropout rate. Defaults to 0.1.
            use_quick_gelu (bool): Whether to use quickgelu instead of gelu. Defaults to False.
            do_classification (bool): Whether to include the final classification head. Defaults to True.
            dtype (DTypeLike): Data type for computations. Defaults to jnp.float32.
            param_dtype (DTypeLike): Data type for parameters. Defaults to jnp.float32.
            rngs (nnx.Rngs): Random number generator keys. Defaults to nnx.Rngs(0).
            mesh (Mesh|None): Optional JAX device mesh for parameter sharding. Defaults to None.
        """
        self.do_classification = do_classification
        self.encoder = VisionTransformerBase(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout_rate=dropout_rate,
            use_quick_gelu=use_quick_gelu,
            use_pre_norm=False,
            use_patch_bias=True,
            layernorm_epsilon=1e-12,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            mesh=mesh,
        )

        if self.do_classification:
            self.classifier = nnx.Linear(
                hidden_size,
                num_classes,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
                kernel_init=sharded_init(nnx.initializers.xavier_uniform(), P(None, "model"), mesh),
                bias_init=sharded_init(nnx.initializers.zeros_init(), P("model"), mesh),
            )

        if mesh:
            with mesh:
                shard_model(self)

    def __call__(self, x: Float[Array, "batch height width channels"]) -> Float[Array, "batch num_classes"]:
        """Forward pass of the Vision Transformer.

        Args:
            x (Float[Array, "batch height width channels"]): Input tensor with shape [batch, height, width, channels]

        Returns:
            Float[Array, "batch num_classes"]: Output logits with shape [batch, num_classes]
        """
        x = self.encoder(x)
        if self.do_classification:
            return self.classifier(x)
        return x

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, use_pytorch: bool = False, mesh: Mesh | None = None, dtype: DTypeLike = jnp.float32) -> "VisionTransformer":
        """Load a pretrained Vision Transformer from a local path or HuggingFace Hub.

        Args:
            model_name_or_path (str): Path to local weights or HuggingFace model ID.
            use_pytorch (bool): Whether to load from PyTorch weights. Defaults to False.
            mesh (Mesh|None): Optional device mesh for parameter sharding. Defaults to None.
            dtype (DTypeLike): Data type for computations. Defaults to jnp.float32.

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
            flax_base = ("encoder", "transformer", "blocks", "layers", i)
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
        used_hf_keys: Set[str] = set()

        for flax_dst_key_tuple, hf_src_key_tuple in params_name_mapping.items():
            assert flax_dst_key_tuple in flax_model_params_fstate, flax_dst_key_tuple
            hf_src_key_as_string = ".".join(hf_src_key_tuple)
            used_hf_keys.add(hf_src_key_as_string)
            assert hf_src_key_as_string in params_fstate, f"HF key '{hf_src_key_as_string}' (from Flax key {flax_dst_key_tuple}) not found in loaded weights."
            nonvisited.remove(flax_dst_key_tuple)
            src_value: Array = params_fstate[hf_src_key_as_string]

            dst_value_obj = flax_model_params_fstate[flax_dst_key_tuple]
            original_param_sharding = dst_value_obj.value.sharding

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

            sharded_new_value: Array = jax.device_put(src_value, original_param_sharding)
            dst_value_obj.value = sharded_new_value

            assert jnp.allclose(dst_value_obj.value.mean(), src_value.mean()), (dst_value_obj.value.mean(), src_value.mean())

        assert len(nonvisited) == 0, f"Some Flax model parameters were not visited: {nonvisited}"

        leftover_hf_keys = set(params_fstate.keys()) - used_hf_keys
        known_unused_hf_buffer_keys = {
            "text_model.embeddings.position_ids",
            "vision_model.embeddings.position_ids",
        }
        unexpected_leftover_hf_keys = leftover_hf_keys - known_unused_hf_buffer_keys

        assert len(unexpected_leftover_hf_keys) == 0, f"Some unexpected HuggingFace checkpoint parameters were not used: {sorted(list(unexpected_leftover_hf_keys))}"
        nnx.update(model, nnx.from_flat_state(flax_model_params_fstate))

        if mesh:
            with mesh:
                shard_model(model)

        del flax_model_params_fstate
        del params_fstate
        return model
