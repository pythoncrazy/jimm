import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib
from jax.sharding import Mesh
from jaxtyping import Array, DTypeLike, Float, Int

from jimm.common.transformer import Transformer
from jimm.common.utils import DEFAULT_SHARDING, MeshRules
from jimm.common.vit import VisionTransformerBase


class CLIPVisionModel(nnx.Module):
    def __init__(
        self,
        image_resolution: int,
        vision_layers: int,
        vision_width: int,
        vision_patch_size: int,
        transformer_width: int,
        use_gradient_checkpointing: bool = False,
        rngs: rnglib.Rngs = nnx.Rngs(0),
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        mesh: Mesh | None = None,
        mesh_rules: MeshRules = DEFAULT_SHARDING,
    ):
        """Initialize the Vision Encoder with projection.

        Args:
            image_resolution (int): The resolution of the input images.
            vision_layers (int): The number of layers in the vision transformer.
            vision_width (int): The width of the vision transformer.
            vision_patch_size (int): The patch size of the vision transformer.
            transformer_width (int): The output dimension after projection.
            use_gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
            rngs (rnglib.Rngs, optional): The random number generator state. Defaults to nnx.Rngs(0).
            dtype (DTypeLike, optional): The data type for computations. Defaults to jnp.float32.
            param_dtype (DTypeLike, optional): The data type for parameters. Defaults to jnp.float32.
            mesh (Mesh | None, optional): The device mesh for parameter sharding. Defaults to None.
            mesh_rules (MeshRules, optional): Logical axis sharding rules. Defaults to DEFAULT_SHARDING.
        """
        self.vision_layers = vision_layers
        self.vision_width = vision_width
        self.vision_patch_size = vision_patch_size
        self.transformer_width = transformer_width
        self.dtype = dtype

        vision_heads = vision_width // 64

        self.encoder = VisionTransformerBase(
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
            use_gradient_checkpointing=use_gradient_checkpointing,
            pooling_type="CLS",
            layernorm_epsilon=1e-5,
            dtype=dtype,
            param_dtype=param_dtype,
            mesh=mesh,
            rngs=rngs,
            mesh_rules=mesh_rules,
        )
        self.visual_projection = nnx.Linear(
            vision_width,
            transformer_width,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), mesh_rules("visual_proj_in", "visual_proj_out")),
        )

    def __call__(self, image: Float[Array, "batch height width channels"], do_projection: bool = True) -> Float[Array, "batch vision_width_or_transformer_width"]:
        """Encode images into embeddings.

        Args:
            image (Float[Array, "batch height width channels"]): Batch of input images.
            do_projection (bool): Whether to apply the visual projection layer. Defaults to True.

        Returns:
            Float[Array, "batch vision_width_or_transformer_width"]: Image embeddings.
            Shape depends on do_projection: vision_width if False, transformer_width if True.
        """
        features = self.encoder(image)
        if do_projection:
            return self.visual_projection(features)
        return features

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
    ) -> "CLIPVisionModel":
        """Load a pretrained vision encoder from a CLIP checkpoint.

        Args:
            model_name_or_path (str): Path to local weights or HuggingFace model ID.
            use_pytorch (bool): Whether to load from PyTorch weights. Defaults to False.
            mesh (Mesh | None): Optional device mesh for parameter sharding. Defaults to None.
            dtype (DTypeLike): Data type for computations. Defaults to jnp.float32.
            param_dtype (DTypeLike): Data type for parameters. Defaults to jnp.float32.
            use_gradient_checkpointing (bool): Whether to use gradient checkpointing. Defaults to False.
            rngs (rnglib.Rngs): Random number generator keys. Defaults to nnx.Rngs(0).

        Returns:
            CLIPVisionModel: Pretrained CLIP vision model
        """
        from .params import load_vision_from_pretrained

        return load_vision_from_pretrained(cls, model_name_or_path, use_pytorch, mesh, dtype, param_dtype, use_gradient_checkpointing, rngs)


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
        use_gradient_checkpointing: bool = False,
        rngs: rnglib.Rngs = nnx.Rngs(0),
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        mesh: Mesh | None = None,
        mesh_rules: MeshRules = DEFAULT_SHARDING,
    ):
        """Initialize the CLIP model.

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
            use_gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
            rngs (rnglib.Rngs, optional): The random number generator state. Defaults to nnx.Rngs(0).
            dtype (DTypeLike, optional): The data type for computations. Defaults to jnp.float32.
            param_dtype (DTypeLike, optional): The data type for parameters. Defaults to jnp.float32.
            mesh (Mesh | None, optional): The device mesh for parameter sharding. Defaults to None.
            mesh_rules (MeshRules, optional): Logical axis sharding rules. Defaults to DEFAULT_SHARDING.
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
        self._original_config = None

        self.attn_mask: Float[Array, "context_length context_length"] = jnp.tril(jnp.ones((context_length, context_length), dtype=dtype))

        self.vision_model = CLIPVisionModel(
            image_resolution=image_resolution,
            vision_layers=vision_layers,
            vision_width=vision_width,
            vision_patch_size=vision_patch_size,
            transformer_width=transformer_width,
            use_gradient_checkpointing=use_gradient_checkpointing,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            mesh=mesh,
            mesh_rules=mesh_rules,
        )

        self.text_model = Transformer(
            width=transformer_width,
            mlp_dim=transformer_width * 4,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            dropout_rate=0.0,
            attn_mask=self.attn_mask,
            use_quick_gelu=True,
            use_gradient_checkpointing=use_gradient_checkpointing,
            dtype=dtype,
            param_dtype=param_dtype,
            mesh=mesh,
            rngs=rngs,
            mesh_rules=mesh_rules,
        )
        self.vocab_size = vocab_size
        self.token_embedding = nnx.Embed(
            num_embeddings=vocab_size,
            features=transformer_width,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            embedding_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), mesh_rules("token_embed_vocab", "token_embed_hidden")),
        )
        self.positional_embedding = nnx.Param(
            nnx.with_partitioning(nnx.initializers.truncated_normal(stddev=0.02), mesh_rules("pos_embed_seq", "pos_embed_hidden"))(rngs.params(), (context_length, transformer_width))
        )
        self.ln_final = nnx.LayerNorm(
            transformer_width,
            epsilon=1e-5,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            scale_init=nnx.with_partitioning(
                nnx.initializers.ones_init(),
                mesh_rules(
                    "layernorm_dim",
                ),
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros_init(),
                mesh_rules(
                    "layernorm_dim",
                ),
            ),
        )
        self.text_projection = nnx.Linear(
            transformer_width,
            transformer_width,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), mesh_rules("text_proj_in", "text_proj_out")),
        )
        self.logit_scale = nnx.Param(nnx.with_partitioning(nnx.initializers.ones_init(), ())(rngs.params(), ()))

    def encode_image(self, image: Float[Array, "batch height width channels"], do_projection: bool = True) -> Float[Array, "batch transformer_width"]:
        """Encode images into embeddings.

        Args:
            image (Float[Array, "batch height width channels"]): Batch of input images.
            do_projection (bool): Whether the image encoder should do the visual projection layer. Defaults to true.

        Returns:
            Float[Array, "batch transformer_width"]: Image embeddings.
        """
        return self.vision_model(image, do_projection)

    def encode_text(self, text: Int[Array, "batch context_length"]) -> Float[Array, "batch transformer_width"]:
        """Encode text tokens into embeddings.

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
        """Calculate similarity between image and text embeddings.

        Args:
            image (Float[Array, "batch height width channels"]): Batch of input images.
            text (Int[Array, "batch context_length"]): Batch of token sequences.

        Returns:
            Float[Array, "batch batch"]: Similarity scores between all pairs of images and texts.
        """
        image_features: Float[Array, "batch transformer_width"] = self.encode_image(image, do_projection=True)
        text_features: Float[Array, "batch transformer_width"] = self.encode_text(text)

        image_features: Float[Array, "batch transformer_width"] = image_features / jnp.linalg.norm(image_features, axis=-1, keepdims=True)
        text_features: Float[Array, "batch transformer_width"] = text_features / jnp.linalg.norm(text_features, axis=-1, keepdims=True)

        logit_scale: Float[Array, ""] = jnp.exp(self.logit_scale.value)
        logits: Float[Array, "batch batch"] = logit_scale * image_features @ text_features.T
        return logits

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
    ) -> "CLIP":
        """Load a pretrained CLIP model from a local path or HuggingFace Hub.

        Args:
            model_name_or_path (str): Path to local weights or HuggingFace model ID.
            use_pytorch (bool): Whether to load from PyTorch weights. Defaults to False.
            mesh (Mesh | None): Optional device mesh for parameter sharding. Defaults to None.
            dtype (DTypeLike): Data type for computations. Defaults to jnp.float32.
            param_dtype (DTypeLike): Data type for parameters. Defaults to jnp.float32.
            use_gradient_checkpointing (bool): Whether to use gradient checkpointing. Defaults to False.
            rngs (rnglib.Rngs): Random number generator keys. Defaults to nnx.Rngs(0).

        Returns:
            CLIP: Pretrained CLIP model
        """
        from .params import load_from_pretrained

        return load_from_pretrained(cls, model_name_or_path, use_pytorch, mesh, dtype, param_dtype, use_gradient_checkpointing, rngs)

    def save_pretrained(self, save_directory: str) -> None:
        """Save the model weights and config in HuggingFace format.

        Args:
            save_directory (str): Directory path where the model will be saved.
        """
        from .params import save_pretrained

        save_pretrained(self, save_directory)
