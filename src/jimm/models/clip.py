from typing import Optional, Set

import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, DTypeLike, Float, Int

from jimm.common.transformer import Transformer
from jimm.common.utils import sharded_init


# Needed as the CLIP Vision Transformer has an extra layernorm compared to the other vision transformer
class VisionTransformer(nnx.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        num_heads: int,
        output_dim: int,
        rngs: nnx.Rngs = nnx.Rngs(0),
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        mesh: Mesh | None = None,
    ):
        n_patches: int = (input_resolution // patch_size) ** 2
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nnx.Conv(
            in_features=3,
            out_features=width,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            kernel_init=sharded_init(nnx.initializers.xavier_uniform(), P(None, None, None, "model"), mesh),
            bias_init=sharded_init(nnx.initializers.zeros_init(), P("model"), mesh),
        )
        _cls_token_initializer = sharded_init(nnx.initializers.zeros_init(), P(None, None, "model"), mesh)
        cls_token_value: Float[Array, "one one width"] = _cls_token_initializer(rngs.params(), (1, 1, width), dtype=dtype)
        self.cls_token = nnx.Param(cls_token_value)
        _position_embeddings_initializer = sharded_init(nnx.initializers.truncated_normal(stddev=0.02), P(None, None, "model"), mesh)
        pos_emb_value: Float[Array, "one n_patches_plus_1 hidden_size_dim"] = _position_embeddings_initializer(rngs.params(), (1, n_patches + 1, width), dtype=dtype)
        self.position_embeddings = nnx.Param(pos_emb_value)

        self.ln_pre = nnx.LayerNorm(
            width,
            epsilon=1e-5,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            scale_init=sharded_init(nnx.initializers.ones_init(), P("model"), mesh),
            bias_init=sharded_init(nnx.initializers.zeros_init(), P("model"), mesh),
        )
        self.transformer = Transformer(
            width=width,
            mlp_dim=width * 4,
            layers=layers,
            num_heads=num_heads,
            dropout_rate=0.0,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            mesh=mesh,
        )
        self.ln_post = nnx.LayerNorm(
            width,
            epsilon=1e-5,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            scale_init=sharded_init(nnx.initializers.ones_init(), P("model"), mesh),
            bias_init=sharded_init(nnx.initializers.zeros_init(), P("model"), mesh),
        )

        self.proj = nnx.Linear(
            width,
            output_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            kernel_init=sharded_init(nnx.initializers.xavier_uniform(), P(None, "model"), mesh),
        )

    def __call__(self, x: Float[Array, "batch height width channels"]) -> Float[Array, "batch output_dim"]:
        """
        Apply the CLIP vision transformer to input images.

        Args:
            x: Float[Array, "batch height width channels"]
                Batch of input images with shape (batch, height, width, channels).

        Returns:
            Float[Array, "batch output_dim"]
                Batch of output embeddings with shape (batch, output_dim).
        """
        patches: Float[Array, "batch n_patches width"] = self.conv1(x)
        batch_size = patches.shape[0]
        patches = patches.reshape(batch_size, -1, patches.shape[-1])
        cls_token = jnp.tile(self.cls_token.value, [batch_size, 1, 1])
        x: Float[Array, "batch n_patches+1 width"] = jnp.concat([cls_token, patches], axis=1)
        embeddings: Float[Array, "batch n_patches+1 width"] = x + self.position_embeddings.value
        x: Float[Array, "batch n_patches+1 width"] = self.ln_pre(embeddings)
        x: Float[Array, "batch n_patches+1 width"] = self.transformer(x)
        x: Float[Array, "batch n_patches+1 width"] = self.ln_post(x)
        x: Float[Array, "batch output_dim"] = self.proj(x[:, 0])
        return x


class CLIP(nnx.Module):
    def __init__(
        self,
        # Vision
        image_resolution: int,
        vision_layers: int,
        vision_width: int,
        vision_patch_size: int,
        # Text
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        rngs: nnx.Rngs = nnx.Rngs(0),
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        mesh: Mesh | None = None,
    ):
        self.vision_layers = vision_layers
        self.vision_width = vision_width
        self.vision_patch_size = vision_patch_size
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.transformer_width = transformer_width
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers
        self.dtype = dtype

        vision_heads = vision_width // 64

        self.attn_mask = jnp.tril(jnp.ones((context_length, context_length), dtype=jnp.bool_))

        # Vision model
        self.vision_model = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            num_heads=vision_heads,
            output_dim=transformer_width,
            dtype=dtype,
            param_dtype=param_dtype,
            mesh=mesh,
            rngs=rngs,
        )

        # Text model
        self.text_model = Transformer(
            width=transformer_width,
            mlp_dim=transformer_width * 4,
            layers=transformer_layers,
            num_heads=transformer_heads,
            dropout_rate=0.0,
            attn_mask=self.attn_mask,
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
        self.positional_embedding = nnx.Param(sharded_init(nnx.initializers.truncated_normal(stddev=0.02), P("model", None), mesh)(rngs.params(), (context_length, transformer_width), dtype=dtype))
        self.ln_final = nnx.LayerNorm(
            transformer_width,
            epsilon=1e-5,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            scale_init=sharded_init(nnx.initializers.ones_init(), P("model"), mesh),
            bias_init=sharded_init(nnx.initializers.zeros_init(), P("model"), mesh),
        )
        self.text_projection = nnx.Linear(
            transformer_width,
            transformer_width,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            kernel_init=sharded_init(nnx.initializers.xavier_uniform(), P("model", None), mesh),
        )
        self.logit_scale = nnx.Param(sharded_init(nnx.initializers.ones_init(), P("model"), mesh)(rngs.params(), (), dtype=dtype))

    def encode_image(self, image: Float[Array, "batch height width channels"]) -> Float[Array, "batch transformer_width"]:
        """
        Encode images into embeddings.

        Args:
            image: Batch of input images.

        Returns:
            Image embeddings.
        """
        return self.vision_model(image)

    def encode_text(self, text: Int[Array, "batch context_length"]) -> Float[Array, "batch transformer_width"]:
        """
        Encode text tokens into embeddings.

        Args:
            text: Batch of token sequences.

        Returns:
            Text embeddings.
        """
        seq_len = text.shape[1]
        x: Float[Array, "batch context_length transformer_width"] = self.token_embedding(text)
        x: Float[Array, "batch context_length transformer_width"] = x + self.positional_embedding.value[:seq_len]
        x: Float[Array, "batch context_length transformer_width"] = self.text_model(x)
        x: Float[Array, "batch context_length transformer_width"] = self.ln_final(x)

        eot_token_pos = jnp.argmax(text, axis=-1)
        batch_indices = jnp.arange(x.shape[0])
        x: Float[Array, "batch transformer_width"] = x[batch_indices, eot_token_pos] @ self.text_projection.kernel.value
        return x

    def __call__(self, image: Float[Array, "batch height width channels"], text: Int[Array, "batch context_length"]) -> Float[Array, "batch batch"]:
        """
        Calculate similarity between image and text embeddings.

        Args:
            image: Batch of input images.
            text: Batch of token sequences.

        Returns:
            Similarity scores between all pairs of images and texts.
        """
        image_features: Float[Array, "batch transformer_width"] = self.encode_image(image)
        text_features: Float[Array, "batch transformer_width"] = self.encode_text(text)

        image_features = image_features / jnp.linalg.norm(image_features, axis=-1, keepdims=True)
        text_features = text_features / jnp.linalg.norm(text_features, axis=-1, keepdims=True)

        logit_scale = jnp.exp(self.logit_scale.value)
        logits: Float[Array, "batch batch"] = logit_scale * image_features @ text_features.T
        return logits

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, use_pytorch: bool = False, mesh: Optional[Mesh] = None, dtype: DTypeLike = jnp.float32) -> "CLIP":
        """Load a pretrained CLIP model from a local path or HuggingFace Hub.

        Args:
            model_name_or_path: Path to local weights or HuggingFace model ID
            use_pytorch: Whether to load from PyTorch weights
            mesh: Optional device mesh for parameter sharding
            dtype: Data type for computations

        Returns:
            Pretrained CLIP model
        """
        import json
        import os

        import jax
        from huggingface_hub import hf_hub_download
        from safetensors.flax import load_file

        params_fstate = None
        config = None

        if use_pytorch:
            import torch

            if os.path.isdir(model_name_or_path):
                config_file_path = os.path.join(model_name_or_path, "config.json")
                weights_file_path = os.path.join(model_name_or_path, "pytorch_model.bin")
            else:
                config_file_path = hf_hub_download(repo_id=model_name_or_path, filename="config.json")
                weights_file_path = hf_hub_download(repo_id=model_name_or_path, filename="pytorch_model.bin")

            with open(config_file_path, "r") as f:
                config = json.load(f)

            state_dict = torch.load(weights_file_path, map_location="cpu")
            params_fstate = {k: jnp.array(v.numpy()) for k, v in state_dict.items()}

        elif os.path.exists(model_name_or_path) and os.path.isfile(model_name_or_path):
            safetensors_file_to_load = model_name_or_path
            params_fstate = load_file(safetensors_file_to_load)

            config_path = model_name_or_path.replace(".safetensors", "").replace("/model", "") + "/config.json"
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
            else:
                # Attempt to infer config from safetensors if config.json is missing
                text_hidden_size = params_fstate["text_model.embeddings.token_embedding.weight"].shape[1]
                text_max_pos_embed = params_fstate["text_model.embeddings.position_embedding.weight"].shape[0]
                text_vocab_size = params_fstate["text_model.embeddings.token_embedding.weight"].shape[0]
                
                text_num_layers = 0
                for k in params_fstate:
                    if k.startswith("text_model.encoder.layers.") and k.endswith(".self_attn.q_proj.weight"):
                        layer_idx = int(k.split(".")[3])
                        text_num_layers = max(text_num_layers, layer_idx + 1)

                vision_hidden_size = params_fstate["vision_model.embeddings.class_embedding"].shape[0]
                vision_patch_size = params_fstate["vision_model.embeddings.patch_embedding.weight"].shape[2]
                # (sqrt(num_pos_embeddings - 1)) * patch_size
                vision_image_size = int((params_fstate["vision_model.embeddings.position_embedding.weight"].shape[0] - 1) ** 0.5) * vision_patch_size
                
                vision_num_layers = 0
                for k in params_fstate:
                    if k.startswith("vision_model.encoder.layers.") and k.endswith(".self_attn.q_proj.weight"):
                        layer_idx = int(k.split(".")[3])
                        vision_num_layers = max(vision_num_layers, layer_idx + 1)

                config = {
                    "text_config": {
                        "hidden_size": text_hidden_size,
                        "num_attention_heads": text_hidden_size // 64, # Common assumption for CLIP
                        "num_hidden_layers": text_num_layers,
                        "max_position_embeddings": text_max_pos_embed,
                        "vocab_size": text_vocab_size,
                    },
                    "vision_config": {
                        "hidden_size": vision_hidden_size,
                        "num_attention_heads": vision_hidden_size // 64, # Common assumption
                        "num_hidden_layers": vision_num_layers,
                        "image_size": vision_image_size,
                        "patch_size": vision_patch_size,
                    },
                }
        else:
            config_file_path = hf_hub_download(repo_id=model_name_or_path, filename="config.json")
            safetensors_file_to_load = hf_hub_download(repo_id=model_name_or_path, filename="model.safetensors")

            with open(config_file_path, "r") as f:
                config = json.load(f)

            params_fstate = load_file(safetensors_file_to_load)

        if params_fstate is None:
            raise ValueError(f"Could not load parameters from {model_name_or_path}")
        if config is None:
            raise ValueError(f"Could not load config for {model_name_or_path}")

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
            mesh=mesh,
            dtype=dtype,
            param_dtype=dtype,
        )

        flax_model_params_fstate = dict(nnx.to_flat_state(nnx.state(model, nnx.Param)))

        mapping_list = [
            (("logit_scale",), ("logit_scale",)),
            (("positional_embedding",), ("text_model", "embeddings", "position_embedding", "weight")),
            (("token_embedding", "embedding"), ("text_model", "embeddings", "token_embedding", "weight")),
            (("ln_final", "scale"), ("text_model", "final_layer_norm", "weight")),
            (("ln_final", "bias"), ("text_model", "final_layer_norm", "bias")),
            (("text_projection", "kernel"), ("text_projection", "weight")),
            (("vision_model", "cls_token"), ("vision_model", "embeddings", "class_embedding")),
            (("vision_model", "position_embeddings"), ("vision_model", "embeddings", "position_embedding", "weight")),
            (("vision_model", "conv1", "kernel"), ("vision_model", "embeddings", "patch_embedding", "weight")),
            (("vision_model", "ln_pre", "scale"), ("vision_model", "pre_layrnorm", "weight")),
            (("vision_model", "ln_pre", "bias"), ("vision_model", "pre_layrnorm", "bias")),
            (("vision_model", "ln_post", "scale"), ("vision_model", "post_layernorm", "weight")),
            (("vision_model", "ln_post", "bias"), ("vision_model", "post_layernorm", "bias")),
            (("vision_model", "proj", "kernel"), ("visual_projection", "weight")),
        ]

        for i in range(text_config["num_hidden_layers"]):
            flax_base = ("text_model", "blocks", "layers", i)
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
            flax_base = ("vision_model", "transformer", "blocks", "layers", i)
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
            if flax_dst_key_tuple not in flax_model_params_fstate:
                continue

            hf_src_key_as_string = ".".join(hf_src_key_tuple)
            if hf_src_key_as_string not in params_fstate:
                continue
            
            used_hf_keys.add(hf_src_key_as_string)
            nonvisited.discard(flax_dst_key_tuple)
            src_value = params_fstate[hf_src_key_as_string]
            dst_value_obj = flax_model_params_fstate[flax_dst_key_tuple]
            original_param_sharding = dst_value_obj.value.sharding

            if flax_dst_key_tuple == ("vision_model", "conv1", "kernel"):
                src_value = jnp.transpose(src_value, (2, 3, 1, 0))
            elif flax_dst_key_tuple == ("vision_model", "cls_token"):
                src_value = src_value.reshape(1, 1, -1)
            elif flax_dst_key_tuple == ("vision_model", "position_embeddings"):
                src_value = src_value.reshape(1, src_value.shape[0], src_value.shape[1])
            elif hf_src_key_tuple[-1] == "weight" and hf_src_key_tuple[-2] in ("q_proj", "k_proj", "v_proj"):
                src_value = jnp.transpose(src_value, (1, 0))
                if flax_dst_key_tuple[0] == "text_model":
                    num_heads = text_config["num_attention_heads"]
                    hidden_size = text_config["hidden_size"]
                else:
                    num_heads = vision_config["hidden_size"] // 64
                    hidden_size = vision_config["hidden_size"]
                head_dim = hidden_size // num_heads
                src_value = src_value.reshape((hidden_size, num_heads, head_dim))
            elif hf_src_key_tuple[-1] == "bias" and hf_src_key_tuple[-2] in ("q_proj", "k_proj", "v_proj"):
                if flax_dst_key_tuple[0] == "text_model":
                    num_heads = text_config["num_attention_heads"]
                    hidden_size = text_config["hidden_size"]
                else:
                    num_heads = vision_config["hidden_size"] // 64
                    hidden_size = vision_config["hidden_size"]
                head_dim = hidden_size // num_heads
                src_value = src_value.reshape((num_heads, head_dim))
            elif hf_src_key_tuple[-2:] == ("out_proj", "weight"):
                src_value = jnp.transpose(src_value, (1, 0))
                if flax_dst_key_tuple[0] == "text_model":
                    num_heads = text_config["num_attention_heads"]
                    hidden_size = text_config["hidden_size"]
                else:
                    num_heads = vision_config["hidden_size"] // 64
                    hidden_size = vision_config["hidden_size"]
                head_dim = hidden_size // num_heads
                src_value = src_value.reshape((num_heads, head_dim, hidden_size))
            elif flax_dst_key_tuple == ("token_embedding", "embedding"):
                pass
            elif flax_dst_key_tuple == ("positional_embedding",):
                pass
            elif hf_src_key_tuple[-1] == "weight" and src_value.ndim == 2:
                src_value = jnp.transpose(src_value, (1, 0))
            
            if src_value.shape != dst_value_obj.value.shape:
                raise ValueError(
                    f"Shape mismatch for {flax_dst_key_tuple} (Flax) vs {hf_src_key_as_string} (HF): "
                    f"{dst_value_obj.value.shape} (expected) != {src_value.shape} (actual)"
                )

            sharded_new_value = jax.device_put(src_value, original_param_sharding)
            dst_value_obj.value = sharded_new_value

        nnx.update(model, nnx.from_flat_state(flax_model_params_fstate))
        assert len(nonvisited) == 0, f"Some Flax CLIP model parameters were not visited: {sorted(list(nonvisited))}"
        
        leftover_hf_keys: Set[str] = hf_checkpoint_keys - used_hf_keys
        known_unused_hf_buffer_keys: Set[str] = {
            "text_model.embeddings.position_ids",
            "vision_model.embeddings.position_ids",
        }
        unexpected_leftover_hf_keys: Set[str] = leftover_hf_keys - known_unused_hf_buffer_keys
        
        assert len(unexpected_leftover_hf_keys) == 0, f"Some unexpected HuggingFace checkpoint parameters were not used: {sorted(list(unexpected_leftover_hf_keys))}"
        
        return model
