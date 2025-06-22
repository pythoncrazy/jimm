import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, DTypeLike, Float

from jimm.common.transformer import Transformer
from jimm.common.utils import sharded_init


class VisionTransformerBase(nnx.Module):
    """A base Vision Transformer (ViT) model."""

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        mlp_dim: int,
        dropout_rate: float = 0.0,
        use_quick_gelu: bool = False,
        use_pre_norm: bool = False,
        use_patch_bias: bool = True,
        layernorm_epsilon: float = 1e-5,
        rngs: nnx.Rngs = nnx.Rngs(0),
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        mesh: Mesh | None = None,
    ):
        """
        Initialize the Vision Transformer base model.

        Args:
            img_size (int): The size of the input images.
            patch_size (int): The patch size of the vision transformer.
            in_channels (int): The number of input channels.
            hidden_size (int): The width of the vision transformer.
            num_layers (int): The number of layers in the vision transformer.
            num_heads (int): The number of attention heads in the vision transformer.
            mlp_dim (int): The dimension of the MLP in the transformer blocks.
            dropout_rate (float): The dropout rate. Defaults to 0.0.
            use_quick_gelu (bool): Whether to use QuickGELU activation. Defaults to False.
            use_pre_norm (bool): Whether to apply LayerNorm before the transformer. Defaults to False.
            use_patch_bias (bool): Whether to use bias in the patch embedding convolution. Defaults to True.
            layernorm_epsilon (float): Epsilon for LayerNorm. Defaults to 1e-5.
            rngs (nnx.Rngs): The random number generator state. Defaults to nnx.Rngs(0).
            dtype (DTypeLike): The data type for computations. Defaults to jnp.float32.
            param_dtype (DTypeLike): The data type for parameters. Defaults to jnp.float32.
            mesh (Mesh | None): The device mesh for parameter sharding.
        """
        n_patches: int = (img_size // patch_size) ** 2
        self.use_pre_norm = use_pre_norm

        self.patch_embeddings = nnx.Conv(
            in_features=in_channels,
            out_features=hidden_size,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            use_bias=use_patch_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            kernel_init=sharded_init(nnx.initializers.xavier_uniform(), P(None, None, None, "model"), mesh),
            bias_init=sharded_init(nnx.initializers.zeros_init(), P("model"), mesh),
        )
        _cls_token_initializer = sharded_init(nnx.initializers.zeros_init(), P(None, None, "model"), mesh)
        cls_token_value: Float[Array, "1 1 hidden_size"] = _cls_token_initializer(rngs.params(), (1, 1, hidden_size))
        self.cls_token = nnx.Param(cls_token_value)

        _position_embeddings_initializer = sharded_init(nnx.initializers.truncated_normal(stddev=0.02), P(None, None, "model"), mesh)
        pos_emb_value: Float[Array, "1 n_patches+1 hidden_size"] = _position_embeddings_initializer(rngs.params(), (1, n_patches + 1, hidden_size))
        self.position_embeddings = nnx.Param(pos_emb_value)

        if self.use_pre_norm:
            self.ln_pre = nnx.LayerNorm(
                hidden_size,
                epsilon=layernorm_epsilon,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
                scale_init=sharded_init(nnx.initializers.ones_init(), P("model"), mesh),
                bias_init=sharded_init(nnx.initializers.zeros_init(), P("model"), mesh),
            )
        else:
            self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

        self.transformer = Transformer(
            width=hidden_size,
            mlp_dim=mlp_dim,
            layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            use_quick_gelu=use_quick_gelu,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            mesh=mesh,
        )

        self.ln_post = nnx.LayerNorm(
            hidden_size,
            epsilon=layernorm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            scale_init=sharded_init(nnx.initializers.ones_init(), P("model"), mesh),
            bias_init=sharded_init(nnx.initializers.zeros_init(), P("model"), mesh),
        )

    def __call__(self, x: Float[Array, "batch height width channels"]) -> Float[Array, "batch hidden_size"]:
        """
        Apply the Vision Transformer to input images.

        Args:
            x: Float[Array, "batch height width channels"]
                Batch of input images.

        Returns:
            Float[Array, "batch hidden_size"]
                Batch of output embeddings from the [CLS] token.
        """
        patches: Float[Array, "batch patches_h patches_w hidden_size"] = self.patch_embeddings(x)
        batch_size = patches.shape[0]
        patches: Float[Array, "batch n_patches hidden_size"] = patches.reshape(batch_size, -1, patches.shape[-1])
        cls_token: Float[Array, "batch 1 hidden_size"] = jnp.tile(self.cls_token.value, [batch_size, 1, 1])
        x: Float[Array, "batch n_patches+1 hidden_size"] = jnp.concat([cls_token, patches], axis=1)
        embeddings: Float[Array, "batch n_patches+1 hidden_size"] = x + self.position_embeddings.value

        if self.use_pre_norm:
            x = self.ln_pre(embeddings)
        else:
            x = self.dropout(embeddings)

        x = self.transformer(x)
        x = self.ln_post(x)
        return x[:, 0]
