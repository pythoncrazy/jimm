from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, DTypeLike, Float, Int

from jimm.common.sharding import ShardingSpec, named_sharding_like, reshard_like, sharding_of
from jimm.common.transformer import Transformer
from jimm.common.vit import VisionTransformerBase
from jimm.models.clip.sharding import CLIPSharding


class CLIPVisionModel(nnx.Module):
    def __init__(
        self,
        image_resolution: int,
        vision_layers: int,
        vision_hidden_size: int,
        vision_patch_size: int,
        projection_dim: int,
        use_gradient_checkpointing: bool = False,
        attention_fn: Callable[..., Any] | None = None,
        rngs: rnglib.Rngs | None = None,
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        sharding: ShardingSpec = CLIPSharding,
    ):
        """Initialize the Vision Encoder with projection.

        Args:
            image_resolution (int): The resolution of the input images.
            vision_layers (int): The number of layers in the vision transformer.
            vision_hidden_size (int): The hidden dimension size of the vision transformer.
            vision_patch_size (int): The patch size of the vision transformer.
            projection_dim (int): The output dimension after projection.
            use_gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
            attention_fn (Callable[..., Any] | None, optional): Custom attention function (e.g. jimm.tokamax_attention or jimm.make_tokamax_attention("mosaic_tpu")). Defaults to None.
            rngs (rnglib.Rngs | None, optional): The random number generator state. If None, initializes to nnx.Rngs(0).
            dtype (DTypeLike, optional): The data type for computations. Defaults to jnp.float32.
            param_dtype (DTypeLike, optional): The data type for parameters. Defaults to jnp.float32.
            sharding (ShardingSpec, optional): Sharding specification for parameters. Defaults to CLIPSharding.
        """
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.vision_layers = vision_layers
        self.vision_hidden_size = vision_hidden_size
        self.vision_patch_size = vision_patch_size
        self.projection_dim = projection_dim
        self.dtype = dtype

        vision_heads = vision_hidden_size // 64

        self.encoder = VisionTransformerBase(
            img_size=image_resolution,
            patch_size=vision_patch_size,
            in_channels=3,
            hidden_size=vision_hidden_size,
            num_layers=vision_layers,
            num_heads=vision_heads,
            mlp_dim=vision_hidden_size * 4,
            use_pre_norm=True,
            use_patch_bias=False,
            use_quick_gelu=True,
            use_gradient_checkpointing=use_gradient_checkpointing,
            attention_fn=attention_fn,
            pooling_type="CLS",
            layernorm_epsilon=1e-5,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            sharding=sharding,
        )
        self.visual_projection = nnx.Linear(
            vision_hidden_size,
            projection_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.xavier_uniform(),
                sharding.proj_kernel,
            ),
        )

    def __call__(self, image: Float[Array, "batch height width channels"], do_projection: bool = True) -> Float[Array, "batch vision_hidden_size_or_projection_dim"]:
        """Encode images into embeddings.

        Args:
            image (Float[Array, "batch height width channels"]): Batch of input images.
            do_projection (bool): Whether to apply the visual projection layer. Defaults to True.

        Returns:
            Float[Array, "batch vision_hidden_size_or_projection_dim"]: Image embeddings.
            Shape depends on do_projection: vision_hidden_size if False, projection_dim if True.
        """
        features = self.encoder(image)
        if do_projection:
            out_shard = named_sharding_like(features, P(sharding_of(features).spec[0], sharding_of(self.visual_projection.kernel[...]).spec[-1]))
            return self.visual_projection(features, out_sharding=out_shard)
        return features

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        use_pytorch: bool = False,
        rngs: rnglib.Rngs | None = None,
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        sharding: ShardingSpec = CLIPSharding,
        use_gradient_checkpointing: bool = False,
        attention_fn: Callable[..., Any] | None = None,
    ) -> "CLIPVisionModel":
        """Load a pretrained vision encoder from a CLIP checkpoint.

        Args:
            model_name_or_path (str): Path to local weights or HuggingFace model ID.
            use_pytorch (bool): Whether to load from PyTorch weights. Defaults to False.
            rngs (rnglib.Rngs | None): Random number generator keys. If None, initializes to nnx.Rngs(0).
            dtype (DTypeLike): Data type for computations. Defaults to jnp.float32.
            param_dtype (DTypeLike): Data type for parameters. Defaults to jnp.float32.
            sharding (ShardingSpec): Sharding specification for parameters. Defaults to CLIPSharding.
            use_gradient_checkpointing (bool): Whether to use gradient checkpointing. Defaults to False.
            attention_fn (Callable[..., Any] | None, optional): Custom attention function (e.g. jimm.tokamax_attention or jimm.make_tokamax_attention("mosaic_tpu")). Defaults to None.

        Returns:
            CLIPVisionModel: Pretrained CLIP vision model
        """
        from .params import load_vision_from_pretrained

        return load_vision_from_pretrained(cls, model_name_or_path, use_pytorch, rngs, dtype, param_dtype, sharding, use_gradient_checkpointing, attention_fn)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        rngs: rnglib.Rngs | None = None,
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        sharding: ShardingSpec = CLIPSharding,
        use_gradient_checkpointing: bool = False,
        attention_fn: Callable[..., Any] | None = None,
    ) -> "CLIPVisionModel":
        """Create model from HuggingFace-compatible config dict.

        Args:
            config (dict[str, Any]): Configuration with "vision_config" and "text_config" keys.
            rngs (rnglib.Rngs | None): Random number generator state. If None, initializes to nnx.Rngs(0).
            dtype (DTypeLike): Data type for computations.
            param_dtype (DTypeLike): Data type for parameters.
            sharding (ShardingSpec): Sharding specification for parameters.
            use_gradient_checkpointing (bool): Enable gradient checkpointing.
            attention_fn (Callable[..., Any] | None): Custom attention function (e.g. jimm.tokamax_attention or jimm.make_tokamax_attention("mosaic_tpu")). Defaults to None.

        Returns:
            CLIPVisionModel: Model with randomly initialized weights.
        """
        if rngs is None:
            rngs = nnx.Rngs(0)
        vision_config = config["vision_config"]
        text_config = config["text_config"]

        return cls(
            image_resolution=vision_config["image_size"],
            vision_layers=vision_config["num_hidden_layers"],
            vision_hidden_size=vision_config["hidden_size"],
            vision_patch_size=vision_config["patch_size"],
            projection_dim=text_config["hidden_size"],
            use_gradient_checkpointing=use_gradient_checkpointing,
            attention_fn=attention_fn,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            sharding=sharding,
        )

    def save_pretrained(self, save_directory: str) -> None:
        """Save model weights and config in HuggingFace format.

        Args:
            save_directory (str): Directory path where the model will be saved.
        """
        from .params import save_vision_pretrained

        save_vision_pretrained(self, save_directory)


class CLIPTextModel(nnx.Module):
    def __init__(
        self,
        context_length: int,
        vocab_size: int,
        text_hidden_size: int,
        num_text_heads: int,
        num_text_layers: int,
        use_gradient_checkpointing: bool = False,
        attention_fn: Callable[..., Any] | None = None,
        rngs: rnglib.Rngs | None = None,
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        sharding: ShardingSpec = CLIPSharding,
    ):
        """Initialize CLIP text encoder.

        Args:
            context_length (int): Maximum sequence length.
            vocab_size (int): Size of vocabulary.
            text_hidden_size (int): Hidden dimension size of the text transformer.
            num_text_heads (int): Number of attention heads in the text transformer.
            num_text_layers (int): Number of transformer layers in the text transformer.
            use_gradient_checkpointing (bool): Enable gradient checkpointing.
            attention_fn (Callable[..., Any] | None, optional): Custom attention function (e.g. jimm.tokamax_attention or jimm.make_tokamax_attention("mosaic_tpu")). Defaults to None.
            rngs (rnglib.Rngs | None): RNG state. If None, initializes to nnx.Rngs(0).
            dtype (DTypeLike): Computation dtype.
            param_dtype (DTypeLike): Parameter dtype.
            sharding (ShardingSpec): Sharding specification for parameters.
        """
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.text_hidden_size = text_hidden_size
        self.num_text_heads = num_text_heads
        self.num_text_layers = num_text_layers
        self.dtype = dtype

        self.token_embedding = nnx.Embed(
            num_embeddings=vocab_size,
            features=text_hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            embedding_init=nnx.with_partitioning(
                nnx.initializers.xavier_uniform(),
                sharding.embed,
            ),
        )
        self.positional_embedding = nnx.Param(
            nnx.with_partitioning(
                nnx.initializers.truncated_normal(stddev=0.02),
                sharding.text_pos_embed,
            )(rngs.params(), (context_length, text_hidden_size))
        )

        attn_mask = jnp.tril(jnp.ones((context_length, context_length), dtype=dtype))
        self.transformer = Transformer(
            hidden_size=text_hidden_size,
            mlp_dim=text_hidden_size * 4,
            num_layers=num_text_layers,
            num_heads=num_text_heads,
            dropout_rate=0.0,
            attn_mask=attn_mask,
            use_quick_gelu=True,
            use_gradient_checkpointing=use_gradient_checkpointing,
            attention_fn=attention_fn,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            sharding=sharding,
        )

        self.ln_final = nnx.LayerNorm(
            text_hidden_size,
            epsilon=1e-5,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            scale_init=nnx.with_partitioning(
                nnx.initializers.ones_init(),
                sharding.layernorm,
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros_init(),
                sharding.layernorm,
            ),
        )

        self.text_projection = nnx.Linear(
            text_hidden_size,
            text_hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.xavier_uniform(),
                sharding.proj_kernel,
            ),
        )

    def __call__(self, text: Int[Array, "batch context_length"], do_projection: bool = True) -> Float[Array, "batch text_hidden_size"]:
        """Encode text tokens into embeddings.

        Args:
            text (Int[Array, "batch context_length"]): Token sequences.
            do_projection (bool): Apply text projection layer.

        Returns:
            Float[Array, "batch text_hidden_size"]: Text embeddings.
        """
        seq_len = text.shape[1]
        text_sharding = sharding_of(text)
        embed_sharding = named_sharding_like(text, P(*text_sharding.spec, None))
        x = self.token_embedding.embedding[...].at[text].get(out_sharding=embed_sharding)
        pos_embed = jnp.broadcast_to(self.positional_embedding[...][:seq_len], x.shape)
        x = x + reshard_like(pos_embed, x)
        x = self.transformer(x)
        x = self.ln_final(x)

        eot_mask = jax.nn.one_hot(jnp.argmax(text, axis=-1), x.shape[1])
        x_spec = sharding_of(x).spec
        pooled_sharding = named_sharding_like(x, P(x_spec[0], x_spec[2]))
        x = jnp.einsum("bsh,bs->bh", x, eot_mask, out_sharding=pooled_sharding)

        if do_projection:
            out_shard = named_sharding_like(x, P(sharding_of(x).spec[0], sharding_of(self.text_projection.kernel[...]).spec[-1]))
            x = self.text_projection(x, out_sharding=out_shard)
        return x

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        use_pytorch: bool = False,
        rngs: rnglib.Rngs | None = None,
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        sharding: ShardingSpec = CLIPSharding,
        use_gradient_checkpointing: bool = False,
        attention_fn: Callable[..., Any] | None = None,
    ) -> "CLIPTextModel":
        """Load pretrained text encoder from CLIP checkpoint.

        Args:
            model_name_or_path (str): Local path or HuggingFace model ID.
            use_pytorch (bool): Load from PyTorch weights.
            rngs (rnglib.Rngs | None): RNG state. If None, initializes to nnx.Rngs(0).
            dtype (DTypeLike): Computation dtype.
            param_dtype (DTypeLike): Parameter dtype.
            sharding (ShardingSpec): Sharding specification for parameters.
            use_gradient_checkpointing (bool): Enable gradient checkpointing.
            attention_fn (Callable[..., Any] | None, optional): Custom attention function (e.g. jimm.tokamax_attention or jimm.make_tokamax_attention("mosaic_tpu")). Defaults to None.

        Returns:
            CLIPTextModel: Pretrained text model.
        """
        from .params import load_text_from_pretrained

        return load_text_from_pretrained(cls, model_name_or_path, use_pytorch, rngs, dtype, param_dtype, sharding, use_gradient_checkpointing, attention_fn)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        rngs: rnglib.Rngs | None = None,
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        sharding: ShardingSpec = CLIPSharding,
        use_gradient_checkpointing: bool = False,
        attention_fn: Callable[..., Any] | None = None,
    ) -> "CLIPTextModel":
        """Create model from HuggingFace-compatible config dict.

        Args:
            config (dict[str, Any]): Configuration with "text_config" key.
            rngs (rnglib.Rngs | None): Random number generator state. If None, initializes to nnx.Rngs(0).
            dtype (DTypeLike): Data type for computations.
            param_dtype (DTypeLike): Data type for parameters.
            sharding (ShardingSpec): Sharding specification for parameters.
            use_gradient_checkpointing (bool): Enable gradient checkpointing.
            attention_fn (Callable[..., Any] | None): Custom attention function (e.g. jimm.tokamax_attention or jimm.make_tokamax_attention("mosaic_tpu")). Defaults to None.

        Returns:
            CLIPTextModel: Model with randomly initialized weights.
        """
        if rngs is None:
            rngs = nnx.Rngs(0)
        text_config = config["text_config"]

        return cls(
            context_length=text_config["max_position_embeddings"],
            vocab_size=text_config["vocab_size"],
            text_hidden_size=text_config["hidden_size"],
            num_text_heads=text_config["num_attention_heads"],
            num_text_layers=text_config["num_hidden_layers"],
            use_gradient_checkpointing=use_gradient_checkpointing,
            attention_fn=attention_fn,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            sharding=sharding,
        )

    def save_pretrained(self, save_directory: str) -> None:
        """Save model weights and config in HuggingFace format.

        Args:
            save_directory (str): Directory path where the model will be saved.
        """
        from .params import save_text_pretrained

        save_text_pretrained(self, save_directory)


class CLIP(nnx.Module):
    def __init__(
        self,
        image_resolution: int,
        vision_layers: int,
        vision_hidden_size: int,
        vision_patch_size: int,
        context_length: int,
        vocab_size: int,
        text_hidden_size: int,
        num_text_heads: int,
        num_text_layers: int,
        use_gradient_checkpointing: bool = False,
        attention_fn: Callable[..., Any] | None = None,
        vision_attention_fn: Callable[..., Any] | None = None,
        text_attention_fn: Callable[..., Any] | None = None,
        rngs: rnglib.Rngs | None = None,
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        sharding: ShardingSpec = CLIPSharding,
    ):
        """Initialize the CLIP model.

        Args:
            image_resolution (int): The resolution of the input images.
            vision_layers (int): The number of layers in the vision transformer.
            vision_hidden_size (int): The hidden dimension size of the vision transformer.
            vision_patch_size (int): The patch size of the vision transformer.
            context_length (int): The maximum sequence length for text.
            vocab_size (int): The size of the vocabulary.
            text_hidden_size (int): The hidden dimension size of the text transformer.
            num_text_heads (int): The number of attention heads in the text transformer.
            num_text_layers (int): The number of layers in the text transformer.
            use_gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
            attention_fn (Callable[..., Any] | None, optional): Custom attention function applied to both encoders. Defaults to None.
            vision_attention_fn (Callable[..., Any] | None, optional): Override attention_fn for the vision encoder only. Defaults to None.
            text_attention_fn (Callable[..., Any] | None, optional): Override attention_fn for the text encoder only. Defaults to None.
            rngs (rnglib.Rngs | None, optional): The random number generator state. If None, initializes to nnx.Rngs(0).
            dtype (DTypeLike, optional): The data type for computations. Defaults to jnp.float32.
            param_dtype (DTypeLike, optional): The data type for parameters. Defaults to jnp.float32.
            sharding (ShardingSpec): Sharding specification for parameters. Defaults to CLIPSharding.
        """
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.vision_layers = vision_layers
        self.vision_hidden_size = vision_hidden_size
        self.vision_patch_size = vision_patch_size
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.text_hidden_size = text_hidden_size
        self.num_text_heads = num_text_heads
        self.num_text_layers = num_text_layers
        self.dtype = dtype
        self._original_config = None

        self.vision_model = CLIPVisionModel(
            image_resolution=image_resolution,
            vision_layers=vision_layers,
            vision_hidden_size=vision_hidden_size,
            vision_patch_size=vision_patch_size,
            projection_dim=text_hidden_size,
            use_gradient_checkpointing=use_gradient_checkpointing,
            attention_fn=vision_attention_fn or attention_fn,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            sharding=sharding,
        )

        self.text_model = CLIPTextModel(
            context_length=context_length,
            vocab_size=vocab_size,
            text_hidden_size=text_hidden_size,
            num_text_heads=num_text_heads,
            num_text_layers=num_text_layers,
            use_gradient_checkpointing=use_gradient_checkpointing,
            attention_fn=text_attention_fn or attention_fn,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            sharding=sharding,
        )
        self.logit_scale = nnx.Param(nnx.with_partitioning(nnx.initializers.ones_init(), ())(rngs.params(), ()))

    def encode_image(self, image: Float[Array, "batch height width channels"], do_projection: bool = True) -> Float[Array, "batch text_hidden_size"]:
        """Encode images into embeddings.

        Args:
            image (Float[Array, "batch height width channels"]): Batch of input images.
            do_projection (bool): Whether the image encoder should do the visual projection layer. Defaults to true.

        Returns:
            Float[Array, "batch text_hidden_size"]: Image embeddings.
        """
        return self.vision_model(image, do_projection)

    def encode_text(self, text: Int[Array, "batch context_length"]) -> Float[Array, "batch text_hidden_size"]:
        """Encode text tokens into embeddings.

        Args:
            text (Int[Array, "batch context_length"]): Batch of token sequences.

        Returns:
            Float[Array, "batch text_hidden_size"]: Text embeddings.
        """
        return self.text_model(text, do_projection=True)

    def __call__(self, image: Float[Array, "batch height width channels"], text: Int[Array, "batch context_length"]) -> Float[Array, "batch batch"]:
        """Calculate similarity between image and text embeddings.

        Args:
            image (Float[Array, "batch height width channels"]): Batch of input images.
            text (Int[Array, "batch context_length"]): Batch of token sequences.

        Returns:
            Float[Array, "batch batch"]: Similarity scores between all pairs of images and texts.
        """
        image_features: Float[Array, "batch text_hidden_size"] = self.encode_image(image, do_projection=True)
        text_features: Float[Array, "batch text_hidden_size"] = self.encode_text(text)

        image_features: Float[Array, "batch text_hidden_size"] = image_features / jnp.linalg.norm(image_features, axis=-1, keepdims=True)
        text_features: Float[Array, "batch text_hidden_size"] = text_features / jnp.linalg.norm(text_features, axis=-1, keepdims=True)

        logit_scale: Float[Array, ""] = jnp.exp(self.logit_scale[...])
        image_spec = sharding_of(image_features).spec
        logits_sharding = named_sharding_like(image_features, P(image_spec[0], None))
        logits: Float[Array, "batch batch"] = logit_scale * jnp.matmul(image_features, text_features.T, out_sharding=logits_sharding)
        return logits

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        use_pytorch: bool = False,
        rngs: rnglib.Rngs | None = None,
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        sharding: ShardingSpec = CLIPSharding,
        use_gradient_checkpointing: bool = False,
        attention_fn: Callable[..., Any] | None = None,
    ) -> "CLIP":
        """Load a pretrained CLIP model from a local path or HuggingFace Hub.

        Args:
            model_name_or_path (str): Path to local weights or HuggingFace model ID.
            use_pytorch (bool): Whether to load from PyTorch weights. Defaults to False.
            rngs (rnglib.Rngs | None): Random number generator keys. If None, initializes to nnx.Rngs(0).
            dtype (DTypeLike): Data type for computations. Defaults to jnp.float32.
            param_dtype (DTypeLike): Data type for parameters. Defaults to jnp.float32.
            sharding (ShardingSpec): Sharding specification for parameters. Defaults to CLIPSharding.
            use_gradient_checkpointing (bool): Whether to use gradient checkpointing. Defaults to False.
            attention_fn (Callable[..., Any] | None, optional): Custom attention function (e.g. jimm.tokamax_attention or jimm.make_tokamax_attention("mosaic_tpu")). Defaults to None.

        Returns:
            CLIP: Pretrained CLIP model
        """
        from .params import load_from_pretrained

        return load_from_pretrained(cls, model_name_or_path, use_pytorch, rngs, dtype, param_dtype, sharding, use_gradient_checkpointing, attention_fn)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        rngs: rnglib.Rngs | None = None,
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        sharding: ShardingSpec = CLIPSharding,
        use_gradient_checkpointing: bool = False,
        attention_fn: Callable[..., Any] | None = None,
        vision_attention_fn: Callable[..., Any] | None = None,
        text_attention_fn: Callable[..., Any] | None = None,
    ) -> "CLIP":
        """Create model from HuggingFace-compatible config dict.

        Args:
            config (dict[str, Any]): Configuration with "text_config" and "vision_config" keys.
            rngs (rnglib.Rngs | None): Random number generator state. If None, initializes to nnx.Rngs(0).
            dtype (DTypeLike): Data type for computations.
            param_dtype (DTypeLike): Data type for parameters.
            sharding (ShardingSpec): Sharding specification for parameters.
            use_gradient_checkpointing (bool): Enable gradient checkpointing.
            attention_fn (Callable[..., Any] | None): Custom attention function applied to both encoders. Defaults to None.
            vision_attention_fn (Callable[..., Any] | None): Override attention_fn for the vision encoder only. Defaults to None.
            text_attention_fn (Callable[..., Any] | None): Override attention_fn for the text encoder only. Defaults to None.

        Returns:
            CLIP: Model with randomly initialized weights.
        """
        if rngs is None:
            rngs = nnx.Rngs(0)
        text_config = config["text_config"]
        vision_config = config["vision_config"]

        return cls(
            image_resolution=vision_config["image_size"],
            vision_layers=vision_config["num_hidden_layers"],
            vision_hidden_size=vision_config["hidden_size"],
            vision_patch_size=vision_config["patch_size"],
            context_length=text_config["max_position_embeddings"],
            vocab_size=text_config["vocab_size"],
            text_hidden_size=text_config["hidden_size"],
            num_text_heads=text_config["num_attention_heads"],
            num_text_layers=text_config["num_hidden_layers"],
            use_gradient_checkpointing=use_gradient_checkpointing,
            attention_fn=attention_fn,
            vision_attention_fn=vision_attention_fn,
            text_attention_fn=text_attention_fn,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            sharding=sharding,
        )

    def save_pretrained(self, save_directory: str) -> None:
        """Save the model weights and config in HuggingFace format.

        Args:
            save_directory (str): Directory path where the model will be saved.
        """
        from .params import save_pretrained

        save_pretrained(self, save_directory)
