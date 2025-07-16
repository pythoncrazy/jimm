from typing import Any, Set

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from jaxtyping import Array, DTypeLike, Float

from jimm.common.utils import load_params_and_config
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
            pooling_type="MAP",
            layernorm_epsilon=1e-6,
            dtype=dtype,
            param_dtype=param_dtype,
            mesh=mesh,
            rngs=rngs,
        )

    def encode_image(self, image: Float[Array, "batch height width channels"]) -> Float[Array, "batch transformer_width"]:
        """
        Encode images into embeddings.

        Args:
            image (Float[Array, "batch height width channels"]): Batch of input images.

        Returns:
            Float[Array, "batch transformer_width"]: Image embeddings.
        """
        return self.vision_model(image)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, use_pytorch: bool = False, mesh: Mesh | None = None, dtype: DTypeLike = jnp.float32) -> "SigLIP":
        """Load a pretrained SigLIP model from a local path or HuggingFace Hub.

        Args:
            model_name_or_path (str): Path to local weights or HuggingFace model ID.
            use_pytorch (bool): Whether to load from PyTorch weights. Defaults to False.
            mesh (Mesh|None): Optional device mesh for parameter sharding. Defaults to None.
            dtype (DTypeLike): Data type for computations. Defaults to jnp.float32.

        Returns:
            SigLIP: Pretrained SigLIP model
        """
        params_fstate, config_dict = load_params_and_config(model_name_or_path, use_pytorch)

        config: dict[str, Any] = config_dict

        vision_patch_size = params_fstate["vision_model.embeddings.patch_embedding.weight"].shape[3]
        vision_width = params_fstate["vision_model.embeddings.patch_embedding.bias"].shape[0]
        vision_layers = 0
        for k in params_fstate:
            if k.startswith("vision_model.encoder.layers.") and k.endswith(".mlp.fc2.bias"):
                vision_layers = max(vision_layers, int(k.split(".")[3]) + 1)

        model = cls(
            image_resolution=config["vision_config"]["image_size"],
            vision_layers=vision_layers,
            vision_width=vision_width,
            vision_patch_size=vision_patch_size,
            context_length=0,
            vocab_size=0,
            transformer_width=0,
            transformer_heads=0,
            transformer_layers=0,
            mesh=mesh,
            dtype=dtype,
            param_dtype=dtype,
        )

        flax_model_params_fstate = dict(nnx.to_flat_state(nnx.state(model, nnx.Param)))
        nonvisited = set(flax_model_params_fstate.keys())
        used_hf_keys: Set[str] = set()

        mapping_list = [
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
        ]

        for i in range(vision_layers):
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

        for flax_dst_key_tuple, hf_src_key_tuple in params_name_mapping.items():
            hf_src_key_as_string = ".".join(hf_src_key_tuple)
            if hf_src_key_as_string not in params_fstate:
                continue
            nonvisited.discard(flax_dst_key_tuple)
            used_hf_keys.add(hf_src_key_as_string)
            src_value = params_fstate[hf_src_key_as_string]
            dst_value_obj = flax_model_params_fstate[flax_dst_key_tuple]
            original_param_sharding = dst_value_obj.value.sharding

            if flax_dst_key_tuple == ("vision_model", "patch_embeddings", "kernel"):
                src_value = jnp.transpose(src_value, (2, 3, 1, 0))
            elif flax_dst_key_tuple == ("vision_model", "position_embeddings"):
                src_value = src_value.reshape(1, src_value.shape[0], src_value.shape[1])
            elif hf_src_key_tuple[-1] == "weight" and hf_src_key_tuple[-2] in ("q_proj", "k_proj", "v_proj"):
                src_value = jnp.transpose(src_value, (1, 0))
                num_heads = model.vision_heads
                head_dim = vision_width // num_heads
                src_value = src_value.reshape((vision_width, num_heads, head_dim))
            elif hf_src_key_tuple[-1] == "bias" and hf_src_key_tuple[-2] in ("q_proj", "k_proj", "v_proj"):
                num_heads = model.vision_heads
                head_dim = vision_width // num_heads
                src_value = src_value.reshape((num_heads, head_dim))
            elif hf_src_key_tuple[-2:] == ("out_proj", "weight"):
                src_value = jnp.transpose(src_value, (1, 0))
                num_heads = model.vision_heads
                head_dim = vision_width // num_heads
                src_value = src_value.reshape((num_heads, head_dim, vision_width))
            elif hf_src_key_tuple[-1] == "weight" and src_value.ndim == 2:
                src_value = jnp.transpose(src_value, (1, 0))

            if src_value.shape != dst_value_obj.value.shape:
                raise ValueError(f"Shape mismatch for {flax_dst_key_tuple} (Flax) vs {hf_src_key_as_string} (HF): {dst_value_obj.value.shape} (expected) != {src_value.shape} (actual)")

            sharded_new_value = jax.device_put(src_value, original_param_sharding)
            dst_value_obj.value = sharded_new_value

        in_proj_weight = params_fstate["vision_model.head.attention.in_proj_weight"]
        in_proj_bias = params_fstate["vision_model.head.attention.in_proj_bias"]
        used_hf_keys.add("vision_model.head.attention.in_proj_weight")
        used_hf_keys.add("vision_model.head.attention.in_proj_bias")

        q_w, k_w, v_w = jnp.split(in_proj_weight, 3, axis=0)
        q_b, k_b, v_b = jnp.split(in_proj_bias, 3, axis=0)

        num_heads = model.vision_heads
        head_dim = vision_width // num_heads

        for key_part, hf_val in [("query", (q_w, q_b)), ("key", (k_w, k_b)), ("value", (v_w, v_b))]:
            w, b = hf_val
            w = jnp.transpose(w, (1, 0)).reshape(vision_width, num_heads, head_dim)
            b = b.reshape(num_heads, head_dim)
            flax_model_params_fstate[("vision_model", "MAPHead", "attn", key_part, "kernel")].value = w
            flax_model_params_fstate[("vision_model", "MAPHead", "attn", key_part, "bias")].value = b
            nonvisited.discard(("vision_model", "MAPHead", "attn", key_part, "kernel"))
            nonvisited.discard(("vision_model", "MAPHead", "attn", key_part, "bias"))

        out_proj_w = params_fstate["vision_model.head.attention.out_proj.weight"]
        out_proj_b = params_fstate["vision_model.head.attention.out_proj.bias"]
        used_hf_keys.add("vision_model.head.attention.out_proj.weight")
        used_hf_keys.add("vision_model.head.attention.out_proj.bias")
        out_proj_w = jnp.transpose(out_proj_w, (1, 0)).reshape(num_heads, head_dim, vision_width)
        flax_model_params_fstate[("vision_model", "MAPHead", "attn", "out", "kernel")].value = out_proj_w
        flax_model_params_fstate[("vision_model", "MAPHead", "attn", "out", "bias")].value = out_proj_b
        nonvisited.discard(("vision_model", "MAPHead", "attn", "out", "kernel"))
        nonvisited.discard(("vision_model", "MAPHead", "attn", "out", "bias"))

        nnx.update(model, nnx.from_flat_state(flax_model_params_fstate))
        return model
