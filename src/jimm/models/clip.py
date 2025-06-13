import jax.numpy as jnp
from flax import nnx
from jaxtyping import DTypeLike
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jimm.common.transformer import Transformer
from jimm.common.utils import sharded_init
from jaxtyping import Array, Float
from typing import Optional


# Needed as the CLIP Vision Transformer has an extra layernorm compared to the other vision transformer
class VisionTransformer(nnx.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
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
            heads=heads,
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
        x: Float[Array, "batch n_patches width"] = self.conv1(x)
        x: Float[Array, "batch n_patches width"] = x + self.position_embeddings
        x: Float[Array, "batch n_patches+1 width"] = jnp.concatenate([self.cls_token, x], axis=1)
        x: Float[Array, "batch n_patches+1 width"] = self.ln_pre(x)
        x: Float[Array, "batch n_patches+1 width"] = self.transformer(x)
        x: Float[Array, "batch n_patches+1 width"] = self.ln_post(x)
        x: Float[Array, "batch n_patches+1 output_dim"] = self.proj(x)
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
            heads=vision_heads,
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
            heads=transformer_heads,
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
        self.positional_embedding = nnx.Param(
            sharded_init(nnx.initializers.truncated_normal(stddev=0.02), P("model", None), mesh)(rngs.params(), (context_length, transformer_width), dtype=dtype)
        )
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
        self.logit_scale = nnx.Param(
            sharded_init(nnx.initializers.ones_init(), P("model"), mesh)(rngs.params(), (), dtype=dtype)
        )

    def encode_image(self, image: Float[Array, "batch height width channels"]) -> Float[Array, "batch vision_width"]:
        return self.vision_model(image)

    def encode_text(self, text: Float[Array, "batch context_length"]) -> Float[Array, "batch transformer_width"]:
        x = self.token_embedding(text)
        x = x + self.positional_embedding
        x = self.text_model(x)
        x = self.ln_final(x)
        x = x @ self.text_projection
        return x
    
    def __call__(self, image: Float[Array, "batch height width channels"], text: Float[Array, "batch context_length"]) -> Float[Array, "batch output_dim"]:
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        logit_scale = self.logit_scale.value
        logits = logit_scale * image_features @ text_features.T
        return logits
    
    def from_pretrained(self, model_name_or_path: str, use_pytorch: bool = False, mesh: Optional[Mesh] = None, dtype: DTypeLike = jnp.float32) -> "CLIP":
        pass