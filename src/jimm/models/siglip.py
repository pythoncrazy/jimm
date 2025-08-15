import json
import os
from typing import Any, Set

import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, DTypeLike, Float, Int
from safetensors.flax import save_file as save_safetensors

from jimm.common.transformer import Transformer
from jimm.common.utils import convert_state_to_hf_format, load_params_and_config, sharded_init
from jimm.common.vit import VisionTransformerBase


class SigLIP(nnx.Module):
    def __init__(
        self,
        image_resolution: int,
        vision_layers: int,
        vision_width: int,
        vision_patch_size: int,
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        use_gradient_checkpointing: bool = False,
        rngs: rnglib.Rngs = nnx.Rngs(0),
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        mesh: Mesh | None = None,
    ):
        """
        Initialize the SigLIP model.

        Args:
            image_resolution (int): The resolution of the input images.
            vision_layers (int): The number of layers in the vision transformer.
            vision_width (int): The width of the vision transformer.
            vision_patch_size (int): The patch size of the vision transformer.
            context_length (int): The length of the context.
            vocab_size (int): The size of the vocabulary.
            transformer_width (int): The width of the transformer.
            transformer_heads (int): The number of attention heads in the transformer.
            transformer_layers (int): The number of layers in the transformer.
            use_gradient_checkpointing (bool): Whether to use gradient checkpointing. Defaults to False.
            rngs (rnglib.Rngs): The random number generator state. Defaults to nnx.Rngs(0).
            dtype (DTypeLike): The data type for computations. Defaults to jnp.float32.
            param_dtype (DTypeLike): The data type for parameters. Defaults to jnp.float32.
            mesh (Mesh | None): Optional device mesh for parameter sharding. Defaults to None.
        """
        self.vision_layers = vision_layers
        self.vision_width = vision_width
        self.vision_patch_size = vision_patch_size
        self.context_length = context_length
        self.transformer_width = transformer_width
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers
        self.dtype = dtype
        self._original_config = None

        self.vision_heads = vision_width // 64
        self.vision_model = VisionTransformerBase(
            img_size=image_resolution,
            patch_size=vision_patch_size,
            in_channels=3,
            hidden_size=vision_width,
            num_layers=vision_layers,
            num_heads=self.vision_heads,
            mlp_dim=vision_width * 4,
            use_pre_norm=False,
            use_patch_bias=True,
            use_quick_gelu=False,
            use_gradient_checkpointing=use_gradient_checkpointing,
            pooling_type="MAP",
            layernorm_epsilon=1e-6,
            dtype=dtype,
            param_dtype=param_dtype,
            mesh=mesh,
            rngs=rngs,
        )

        self.text_model = Transformer(
            width=transformer_width,
            mlp_dim=transformer_width * 4,
            layers=transformer_layers,
            num_heads=transformer_heads,
            dropout_rate=0.0,
            layernorm_epsilon=1e-6,
            use_quick_gelu=False,
            use_gradient_checkpointing=use_gradient_checkpointing,
            dtype=dtype,
            param_dtype=param_dtype,
            mesh=mesh,
            rngs=rngs,
        )
        self.vocab_size = vocab_size
        self.token_embedding = nnx.Embed(
            num_embeddings=vocab_size,
            features=transformer_width,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            embedding_init=sharded_init(nnx.initializers.xavier_uniform(), P("model", None), mesh),
        )
        self.positional_embedding = nnx.Param(sharded_init(nnx.initializers.truncated_normal(stddev=0.02), P("model", None), mesh)(rngs.params(), (context_length, transformer_width)))
        self.ln_final = nnx.LayerNorm(
            transformer_width,
            epsilon=1e-6,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            scale_init=sharded_init(nnx.initializers.ones_init(), P("model"), mesh),
            bias_init=sharded_init(nnx.initializers.zeros_init(), P("model"), mesh),
        )
        self.text_projection = nnx.Linear(
            transformer_width,
            transformer_width,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            kernel_init=sharded_init(nnx.initializers.xavier_uniform(), P("model", None), mesh),
        )
        self.logit_scale = nnx.Param(sharded_init(nnx.initializers.ones_init(), P(), mesh)(rngs.params(), ()))
        self.logit_bias = nnx.Param(sharded_init(nnx.initializers.ones_init(), P(), mesh)(rngs.params(), ()))

    def _create_config(self) -> dict[str, Any]:
        return {
            "model_type": "siglip",
            "text_config": {
                "hidden_size": self.transformer_width,
                "num_attention_heads": self.transformer_heads,
                "num_hidden_layers": self.transformer_layers,
                "max_position_embeddings": self.context_length,
                "vocab_size": self.vocab_size,
            },
            "vision_config": {
                "hidden_size": self.vision_width,
                "num_attention_heads": self.vision_width // 64,
                "num_hidden_layers": self.vision_layers,
                "image_size": self.vision_model.img_size,
                "patch_size": self.vision_patch_size,
            },
        }

    def encode_image(self, image: Float[Array, "batch height width channels"]) -> Float[Array, "batch transformer_width"]:
        """
        Encode images into embeddings.

        Args:
            image (Float[Array, "batch height width channels"]): Batch of input images.

        Returns:
            Float[Array, "batch transformer_width"]: Image embeddings.
        """
        return self.vision_model(image)

    def encode_text(self, text: Int[Array, "batch context_length"]) -> Float[Array, "batch transformer_width"]:
        """
        Encode text tokens into embeddings.

        Args:
            text (Int[Array, "batch context_length"]): Batch of token sequences.

        Returns:
            Float[Array, "batch transformer_width"]: Text embeddings.
        """
        seq_len = text.shape[1]
        x: Float[Array, "batch context_length transformer_width"] = self.token_embedding(text)
        x: Float[Array, "batch context_length transformer_width"] = x + self.positional_embedding.value[:seq_len]
        x: Float[Array, "batch context_length transformer_width"] = self.text_model(x)
        x: Float[Array, "batch context_length transformer_width"] = self.ln_final(x)

        pooled_output = x[:, -1, :]
        x: Float[Array, "batch transformer_width"] = self.text_projection(pooled_output)
        return x

    def __call__(self, image: Float[Array, "batch height width channels"], text: Int[Array, "batch context_length"]) -> Float[Array, "batch batch"]:
        """
        Calculate similarity between image and text embeddings.

        Args:
            image (Float[Array, "batch height width channels"]): Batch of input images.
            text (Int[Array, "batch context_length"]): Batch of token sequences.

        Returns:
            Float[Array, "batch batch"]: Similarity scores between all pairs of images and texts.
        """
        image_features: Float[Array, "batch transformer_width"] = self.encode_image(image)
        text_features: Float[Array, "batch transformer_width"] = self.encode_text(text)

        image_features: Float[Array, "batch transformer_width"] = image_features / jnp.linalg.norm(image_features, axis=-1, keepdims=True)
        text_features: Float[Array, "batch transformer_width"] = text_features / jnp.linalg.norm(text_features, axis=-1, keepdims=True)

        logit_scale: Float[Array, ""] = jnp.exp(self.logit_scale.value)
        logits: Float[Array, "batch batch"] = logit_scale * image_features @ text_features.T + self.logit_bias.value
        return logits

    def save_pretrained(self, save_directory: str):
        _SPECIAL_MAPPINGS = {
            "ln_final.weight": "text_model.final_layer_norm.weight",
            "ln_final.bias": "text_model.final_layer_norm.bias",
            "vision_model.ln_pre.weight": "vision_model.pre_layrnorm.weight",
            "vision_model.ln_pre.bias": "vision_model.pre_layrnorm.bias",
            "vision_model.ln_post.weight": "vision_model.post_layernorm.weight",
            "vision_model.ln_post.bias": "vision_model.post_layernorm.bias",
            "vision_model.cls_token": "vision_model.embeddings.class_embedding",
            "vision_model.position_embeddings": "vision_model.embeddings.position_embedding.weight",
            "vision_model.patch_embeddings.weight": "vision_model.embeddings.patch_embedding.weight",
            "vision_model.patch_embeddings.bias": "vision_model.embeddings.patch_embedding.bias",
            "positional_embedding": "text_model.embeddings.position_embedding.weight",
            "text_position_ids": "text_model.embeddings.position_ids",
            "token_embedding.embedding": "text_model.embeddings.token_embedding.weight",
            "text_projection.weight": "text_model.head.weight",
            "text_projection.bias": "text_model.head.bias",
            "visual_projection.weight": "visual_projection.weight",
            # MAPHead mappings
            "vision_model.MAPHead.probe": "vision_model.head.probe",
            "vision_model.MAPHead.layernorm.weight": "vision_model.head.layernorm.weight",
            "vision_model.MAPHead.layernorm.bias": "vision_model.head.layernorm.bias",
            "vision_model.MAPHead.mlp.fc1.weight": "vision_model.head.mlp.fc1.weight",
            "vision_model.MAPHead.mlp.fc1.bias": "vision_model.head.mlp.fc1.bias",
            "vision_model.MAPHead.mlp.layers.2.weight": "vision_model.head.mlp.fc2.weight",
            "vision_model.MAPHead.mlp.layers.2.bias": "vision_model.head.mlp.fc2.bias",
            "vision_model.MAPHead.attn.in_proj_weight": "vision_model.head.attention.in_proj_weight",
            "vision_model.MAPHead.attn.in_proj_bias": "vision_model.head.attention.in_proj_bias",
            "vision_model.MAPHead.self_attn.out_proj.weight": "vision_model.head.attention.out_proj.weight",
            "vision_model.MAPHead.self_attn.out_proj.bias": "vision_model.head.attention.out_proj.bias",
        }
        _SPECIAL_RENAMINGS = {
            "text_model.layers": "text_model.encoder.layers",
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

        config = self._original_config.copy() if self._original_config else self._create_config()
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        _, state = nnx.split(self)
        state_dict = nnx.to_pure_dict(state)

        # Combine Q, K, V projections for MAPHead attention before conversion
        if "vision_model" in state_dict and "MAPHead" in state_dict["vision_model"]:
            maphead = state_dict["vision_model"]["MAPHead"]
            if "attn" in maphead:
                attn = maphead["attn"]

                # Combine weights: [q_weight, k_weight, v_weight] -> in_proj_weight
                if all(k in attn for k in ["query", "key", "value"]):
                    q_weight = attn["query"]["kernel"]  # shape: (vision_width, num_heads, head_dim)
                    k_weight = attn["key"]["kernel"]
                    v_weight = attn["value"]["kernel"]

                    # Reshape to (vision_width, vision_width) and concatenate
                    q_flat = q_weight.reshape(q_weight.shape[0], -1).T  # (vision_width, vision_width)
                    k_flat = k_weight.reshape(k_weight.shape[0], -1).T
                    v_flat = v_weight.reshape(v_weight.shape[0], -1).T

                    in_proj_weight = jnp.concatenate([q_flat, k_flat, v_flat], axis=0)  # (3*vision_width, vision_width)

                    # Combine biases: [q_bias, k_bias, v_bias] -> in_proj_bias
                    q_bias = attn["query"]["bias"]  # shape: (num_heads, head_dim)
                    k_bias = attn["key"]["bias"]
                    v_bias = attn["value"]["bias"]

                    q_bias_flat = q_bias.flatten()
                    k_bias_flat = k_bias.flatten()
                    v_bias_flat = v_bias.flatten()

                    in_proj_bias = jnp.concatenate([q_bias_flat, k_bias_flat, v_bias_flat], axis=0)  # (3*vision_width,)

                    del attn["query"]
                    del attn["key"]
                    del attn["value"]

                    attn["in_proj_weight"] = in_proj_weight
                    attn["in_proj_bias"] = in_proj_bias

                # Handle out_proj: flatten from (num_heads, head_dim, vision_width) to (vision_width, vision_width)
                if "out" in attn:
                    out_weight = attn["out"]["kernel"]  # shape: (num_heads, head_dim, vision_width)
                    out_bias = attn["out"]["bias"]  # shape: (vision_width,)

                    # Flatten the out_proj weight to (vision_width, vision_width)
                    out_weight_flat = out_weight.reshape(-1, out_weight.shape[-1])  # (num_heads*head_dim, vision_width)

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

        hf_state.pop("vision_model.vision_position_ids", None)

        save_safetensors(hf_state, os.path.join(save_directory, "model.safetensors"))

    @classmethod
    def from_pretrained(
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
            (("vision_model", "patch_embeddings", "kernel"), ("vision_model", "embeddings", "patch_embedding", "weight")),
            (("vision_model", "patch_embeddings", "bias"), ("vision_model", "embeddings", "patch_embedding", "bias")),
            (("vision_model", "position_embeddings"), ("vision_model", "embeddings", "position_embedding", "weight")),
            (("vision_model", "ln_post", "scale"), ("vision_model", "post_layernorm", "weight")),
            (("vision_model", "ln_post", "bias"), ("vision_model", "post_layernorm", "bias")),
            (("vision_model", "MAPHead", "probe"), ("vision_model", "head", "probe")),
            (("vision_model", "MAPHead", "layernorm", "scale"), ("vision_model", "head", "layernorm", "weight")),
            (("vision_model", "MAPHead", "layernorm", "bias"), ("vision_model", "head", "layernorm", "bias")),
            (("vision_model", "MAPHead", "mlp", "layers", 0, "kernel"), ("vision_model", "head", "mlp", "fc1", "weight")),
            (("vision_model", "MAPHead", "mlp", "layers", 0, "bias"), ("vision_model", "head", "mlp", "fc1", "bias")),
            (("vision_model", "MAPHead", "mlp", "layers", 2, "kernel"), ("vision_model", "head", "mlp", "fc2", "weight")),
            (("vision_model", "MAPHead", "mlp", "layers", 2, "bias"), ("vision_model", "head", "mlp", "fc2", "bias")),
            (("vision_model", "MAPHead", "attn", "query", "kernel"), ("vision_model", "head", "attention", "in_proj_weight")),
            (("vision_model", "MAPHead", "attn", "query", "bias"), ("vision_model", "head", "attention", "in_proj_bias")),
            (("vision_model", "MAPHead", "attn", "key", "kernel"), ("vision_model", "head", "attention", "in_proj_weight")),
            (("vision_model", "MAPHead", "attn", "key", "bias"), ("vision_model", "head", "attention", "in_proj_bias")),
            (("vision_model", "MAPHead", "attn", "value", "kernel"), ("vision_model", "head", "attention", "in_proj_weight")),
            (("vision_model", "MAPHead", "attn", "value", "bias"), ("vision_model", "head", "attention", "in_proj_bias")),
            (("vision_model", "MAPHead", "attn", "out", "kernel"), ("vision_model", "head", "attention", "out_proj", "weight")),
            (("vision_model", "MAPHead", "attn", "out", "bias"), ("vision_model", "head", "attention", "out_proj", "bias")),
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
            flax_base = ("vision_model", "encoder", "layers", i)
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

            if flax_dst_key_tuple == ("vision_model", "patch_embeddings", "kernel"):
                src_value = jnp.transpose(src_value, (2, 3, 1, 0))
            elif flax_dst_key_tuple == ("vision_model", "position_embeddings"):
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
