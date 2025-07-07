from typing import Any, Set

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, DTypeLike, Float, Int

from jimm.common.transformer import Transformer
from jimm.common.utils import load_params_and_config, sharded_init
from jimm.common.vit import VisionTransformerBase


class CLIP(nnx.Module):
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
        rngs: nnx.Rngs = nnx.Rngs(0),
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        mesh: Mesh | None = None,
    ):
        """
        Initialize the CLIP model.

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
            rngs (nnx.Rngs): The random number generator state. Defaults to nnx.Rngs(0).
            dtype (DTypeLike): The data type for computations. Defaults to jnp.float32.
            param_dtype (DTypeLike): The data type for parameters. Defaults to jnp.float32.
            mesh (Mesh | None): The device mesh for parameter sharding. Defaults to None.
        """
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

        self.attn_mask: Float[Array, "context_length context_length"] = jnp.tril(jnp.ones((context_length, context_length), dtype=dtype))

        self.vision_model = VisionTransformerBase(
            img_size=image_resolution,
            patch_size=vision_patch_size,
            in_channels=3,
            hidden_size=vision_width,
            num_layers=vision_layers,
            num_heads=vision_heads,
            mlp_dim=vision_width * 4,
            use_pre_norm=True,
            use_patch_bias=False,
            use_quick_gelu=True,
            layernorm_epsilon=1e-5,
            dtype=dtype,
            param_dtype=param_dtype,
            mesh=mesh,
            rngs=rngs,
        )
        self.visual_projection = nnx.Linear(
            vision_width,
            transformer_width,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            kernel_init=sharded_init(nnx.initializers.xavier_uniform(), P(None, "model"), mesh),
        )

        self.text_model = Transformer(
            width=transformer_width,
            mlp_dim=transformer_width * 4,
            layers=transformer_layers,
            num_heads=transformer_heads,
            dropout_rate=0.0,
            attn_mask=self.attn_mask,
            use_quick_gelu=True,
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
        self.logit_scale = nnx.Param(sharded_init(nnx.initializers.ones_init(), P("model"), mesh)(rngs.params(), ()))

    def encode_image(self, image: Float[Array, "batch height width channels"]) -> Float[Array, "batch transformer_width"]:
        """
        Encode images into embeddings.

        Args:
            image (Float[Array, "batch height width channels"]): Batch of input images.

        Returns:
            Float[Array, "batch transformer_width"]: Image embeddings.
        """
        features = self.vision_model(image)
        return self.visual_projection(features)

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

        eot_token_pos: Float[Array, " batch "] = jnp.argmax(text, axis=-1)
        batch_indices: Float[Array, " batch "] = jnp.arange(x.shape[0])
        x: Float[Array, "batch transformer_width"] = x[batch_indices, eot_token_pos] @ self.text_projection.kernel.value
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
        logits: Float[Array, "batch batch"] = logit_scale * image_features @ text_features.T
        return logits

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, use_pytorch: bool = False, mesh: Mesh | None = None, dtype: DTypeLike = jnp.float32) -> "CLIP":
        """Load a pretrained CLIP model from a local path or HuggingFace Hub.

        Args:
            model_name_or_path (str): Path to local weights or HuggingFace model ID.
            use_pytorch (bool): Whether to load from PyTorch weights. Defaults to False.
            mesh (Mesh|None): Optional device mesh for parameter sharding. Defaults to None.
            dtype (DTypeLike): Data type for computations. Defaults to jnp.float32.

        Returns:
            CLIP: Pretrained CLIP model
        """

        params_fstate, config_dict = load_params_and_config(model_name_or_path, use_pytorch)

        config: dict[str, Any] | None = config_dict

        if config is None:
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
            (("vision_model", "patch_embeddings", "kernel"), ("vision_model", "embeddings", "patch_embedding", "weight")),
            (("vision_model", "ln_pre", "scale"), ("vision_model", "pre_layrnorm", "weight")),
            (("vision_model", "ln_pre", "bias"), ("vision_model", "pre_layrnorm", "bias")),
            (("vision_model", "ln_post", "scale"), ("vision_model", "post_layernorm", "weight")),
            (("vision_model", "ln_post", "bias"), ("vision_model", "post_layernorm", "bias")),
            (("visual_projection", "kernel"), ("visual_projection", "weight")),
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

            if flax_dst_key_tuple == ("vision_model", "patch_embeddings", "kernel"):
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
                raise ValueError(f"Shape mismatch for {flax_dst_key_tuple} (Flax) vs {hf_src_key_as_string} (HF): {dst_value_obj.value.shape} (expected) != {src_value.shape} (actual)")

            sharded_new_value = jax.device_put(src_value, original_param_sharding)
            dst_value_obj.value = sharded_new_value

        nnx.update(model, nnx.from_flat_state(flax_model_params_fstate))
        assert len(nonvisited) == 0, f"Some Flax CLIP model parameters were not visited: {sorted(list(nonvisited))}"

        leftover_hf_keys = hf_checkpoint_keys - used_hf_keys
        known_unused_hf_buffer_keys = {
            "text_model.embeddings.position_ids",
            "vision_model.embeddings.position_ids",
        }
        unexpected_leftover_hf_keys = leftover_hf_keys - known_unused_hf_buffer_keys

        assert len(unexpected_leftover_hf_keys) == 0, f"Some unexpected HuggingFace checkpoint parameters were not used: {sorted(list(unexpected_leftover_hf_keys))}"

        return model
