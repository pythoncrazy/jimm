from typing import Optional

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
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            kernel_init=sharded_init(nnx.initializers.xavier_uniform(), P(None, "model"), mesh),
            bias_init=sharded_init(nnx.initializers.zeros_init(), P("model"), mesh),
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
        x: Float[Array, "batch output_dim"] = self.proj(x)
        return x[:, 0]


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
            output_dim=vision_width,
            dtype=dtype,
            param_dtype=param_dtype,
            mesh=mesh,
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
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            kernel_init=sharded_init(nnx.initializers.xavier_uniform(), P("model", None), mesh),
            bias_init=sharded_init(nnx.initializers.zeros_init(), P("model"), mesh),
        )
        self.logit_scale = nnx.Param(sharded_init(nnx.initializers.ones_init(), P("model"), mesh)(rngs.params(), (), dtype=dtype))

    def encode_image(self, image: Float[Array, "batch height width channels"]) -> Float[Array, "batch vision_width"]:
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
        x: Float[Array, "batch context_length transformer_width"] = self.token_embedding(text)
        x: Float[Array, "batch context_length transformer_width"] = x + self.positional_embedding.value
        x: Float[Array, "batch context_length transformer_width"] = self.text_model(x)
        x: Float[Array, "batch context_length transformer_width"] = self.ln_final(x)
        x: Float[Array, "batch transformer_width"] = x[:, 0] @ self.text_projection.kernel
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
        image_features: Float[Array, "batch vision_width"] = self.encode_image(image)
        text_features: Float[Array, "batch transformer_width"] = self.encode_text(text)
        logit_scale = self.logit_scale.value
        logits: Float[Array, "batch batch"] = logit_scale * image_features @ text_features.T
        return logits

    def from_pretrained(self, model_name_or_path: str, use_pytorch: bool = False, mesh: Optional[Mesh] = None, dtype: DTypeLike = jnp.float32) -> "CLIP":
        """
        Load pretrained CLIP model.

        Args:
            model_name_or_path: Path to model or model identifier.
            use_pytorch: Whether to load from PyTorch weights.
            mesh: Optional device mesh for parameter sharding.
            dtype: Data type for model parameters.

        Returns:
            Pretrained CLIP model.
        """
        pass


if __name__ == "__main__":
    import jax
    from jax import random

    rng = random.PRNGKey(0)
    rngs = nnx.Rngs(rng)

    clip = CLIP(
        image_resolution=224,
        vision_layers=12,
        vision_width=768,
        vision_patch_size=16,
        context_length=77,
        vocab_size=49408,
        transformer_width=768,
        transformer_heads=8,
        transformer_layers=12,
        rngs=rngs,
    )
    image = jax.random.normal(rngs.params(), (1, 224, 224, 3), dtype=jnp.float32)
    text = jax.random.randint(rngs.params(), (1, 77), 0, 49408, dtype=jnp.int32)
    logits = clip(image, text)
    print(logits.shape)
    print(logits)
