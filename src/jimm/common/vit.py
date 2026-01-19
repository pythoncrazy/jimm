import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib
from jax.sharding import Mesh
from jaxtyping import Array, DTypeLike, Float

from jimm.common.splash_attention import SplashAttentionConfig, create_splash_attention_fn
from jimm.common.transformer import Transformer
from jimm.common.utils import DEFAULT_SHARDING, MeshRules


class MultiHeadAttentionPoolingHead(nnx.Module):
    """Multihead Attention Pooling, as needed by the SigLIP model"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        layernorm_epsilon: float = 1e-6,
        splash_attention_config: SplashAttentionConfig | None = None,
        rngs: rnglib.Rngs | None = None,
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        mesh: Mesh | None = None,
        mesh_rules: MeshRules = DEFAULT_SHARDING,
    ):
        """Initialization of the Multihead Attention Pooling.

        Args:
            hidden_size (int): The size of the hidden layer, which determines the dimensionality of the model's internal representations.
            intermediate_size (int): The dimension of the intermediate MLP at the end of the MAP head.
            num_heads (int): The number of attention heads.
            layernorm_epsilon (float, optional): The epsilon used in the layernorm. Defaults to 1e-6.
            splash_attention_config (SplashAttentionConfig | None, optional): Configuration for TPU splash attention. Defaults to None.
            rngs (rnglib.Rngs | None, optional): The flax nnx rng to use for initialization. If None, initializes to nnx.Rngs(0).
            dtype (DTypeLike, optional): The data type for computations. Defaults to jnp.float32.
            param_dtype (DTypeLike, optional): The data type for parameters. Defaults to jnp.float32.
            mesh (Mesh | None, optional): The device mesh to use for the proper sharding. Defaults to None.
            mesh_rules (MeshRules, optional): Logical axis sharding rules. Defaults to DEFAULT_SHARDING.
        """
        if rngs is None:
            rngs = nnx.Rngs(0)
        probe_value: Float[Array, "1 1 hidden_size"] = nnx.initializers.zeros_init()(rngs.params(), (1, 1, hidden_size))
        self.probe = nnx.Param(probe_value, sharding_names=mesh_rules("probe_token_batch", "probe_token_seq", "probe_token_hidden"))

        attention_fn = (
            create_splash_attention_fn(
                splash_attention_config,
                num_heads=num_heads,
                head_dim=hidden_size // num_heads,
            )
            if splash_attention_config is not None
            else nnx.dot_product_attention
        )

        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_size,
            broadcast_dropout=False,
            decode=False,
            deterministic=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), mesh_rules("map_attn_in", "map_attn_out")),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros_init(),
                mesh_rules(
                    "map_attn_out",
                ),
            ),
            attention_fn=attention_fn,
        )

        self.layernorm = nnx.LayerNorm(
            num_features=hidden_size,
            epsilon=layernorm_epsilon,
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

        self.mlp = nnx.Sequential(
            nnx.Linear(
                hidden_size,
                intermediate_size,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
                kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), mesh_rules("map_mlp_in", "map_mlp_out")),
                bias_init=nnx.with_partitioning(
                    nnx.initializers.zeros_init(),
                    mesh_rules(
                        "map_mlp_out",
                    ),
                ),
            ),
            nnx.gelu,
            nnx.Linear(
                intermediate_size,
                hidden_size,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
                kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), mesh_rules("map_mlp_out", "map_mlp_in")),
                bias_init=nnx.with_partitioning(
                    nnx.initializers.zeros_init(),
                    mesh_rules(
                        "map_mlp_in",
                    ),
                ),
            ),
        )

    def __call__(self, hidden_state: Float[Array, "batch length hidden_size"]) -> Array:
        """The forward pass of the MAP head.

        Args:
            hidden_state (Float[Array, "batch length hidden_size"]):
        Returns:
            Array: Float[Array, "batch hidden_size"]
        """
        batch_size = hidden_state.shape[0]
        probe: Float[Array, "batch 1 hidden_size"] = jnp.tile(self.probe[...], [batch_size, 1, 1])
        x: Float[Array, "batch 1 hidden_size"] = self.attn(probe, hidden_state, hidden_state, decode=False)
        residual = x
        x: Float[Array, "batch 1 hidden_size"] = self.layernorm(x)
        x = residual + self.mlp(x)
        return x[:, 0]


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
        pooling_type: str = "CLS",
        dropout_rate: float = 0.0,
        use_quick_gelu: bool = False,
        use_pre_norm: bool = False,
        use_patch_bias: bool = True,
        use_gradient_checkpointing: bool = False,
        layernorm_epsilon: float = 1e-5,
        splash_attention_config: SplashAttentionConfig | None = None,
        rngs: rnglib.Rngs | None = None,
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        mesh: Mesh | None = None,
        mesh_rules: MeshRules = DEFAULT_SHARDING,
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
            pooling_type (str, optional): The pooling method, either CLS or MAP. Defaults to "CLS".
            dropout_rate (float, optional): The dropout rate. Defaults to 0.0.
            use_quick_gelu (bool, optional): Whether to use QuickGELU activation. Defaults to False.
            use_pre_norm (bool, optional): Whether to apply LayerNorm before the transformer. Defaults to False.
            use_patch_bias (bool, optional): Whether to use bias in the patch embedding convolution. Defaults to True.
            use_gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
            layernorm_epsilon (float, optional): Epsilon for LayerNorm. Defaults to 1e-5.
            splash_attention_config (SplashAttentionConfig | None, optional): Configuration for TPU splash attention. Defaults to None.
            rngs (rnglib.Rngs | None, optional): The random number generator state. If None, initializes to nnx.Rngs(0).
            dtype (DTypeLike, optional): The data type for computations. Defaults to jnp.float32.
            param_dtype (DTypeLike, optional): The data type for parameters. Defaults to jnp.float32.
            mesh (Mesh | None, optional): The device mesh for parameter sharding. Defaults to None.
            mesh_rules (MeshRules, optional): Logical axis sharding rules. Defaults to DEFAULT_SHARDING.
        """
        if rngs is None:
            rngs = nnx.Rngs(0)
        n_patches: int = (img_size // patch_size) ** 2
        self.use_pre_norm = use_pre_norm
        self.pooling_type = pooling_type

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
            kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), mesh_rules("patch_conv_h", "patch_conv_w", "patch_conv_c", "patch_conv_out")),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros_init(),
                mesh_rules(
                    "patch_conv_out",
                ),
            ),
        )
        if self.pooling_type == "CLS":
            cls_token_value: Float[Array, "1 1 hidden_size"] = nnx.initializers.zeros_init()(rngs.params(), (1, 1, hidden_size))
            self.cls_token = nnx.Param(cls_token_value, sharding_names=mesh_rules("cls_token_batch", "cls_token_seq", "cls_token_hidden"))
            pos_emb_value: Float[Array, "1 n_patches+1 hidden_size"] = nnx.initializers.truncated_normal(stddev=0.02)(rngs.params(), (1, n_patches + 1, hidden_size))
        elif self.pooling_type == "MAP":
            pos_emb_value: Float[Array, "1 n_patches hidden_size"] = nnx.initializers.truncated_normal(stddev=0.02)(rngs.params(), (1, n_patches, hidden_size))
            self.MAPHead = MultiHeadAttentionPoolingHead(
                hidden_size=hidden_size,
                intermediate_size=4 * hidden_size,
                num_heads=num_heads,
                layernorm_epsilon=layernorm_epsilon,
                splash_attention_config=splash_attention_config,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
                mesh=mesh,
                mesh_rules=mesh_rules,
            )
        else:
            raise ValueError("pooling_type must be either MAP or CLS.")
        self.position_embeddings = nnx.Param(pos_emb_value, sharding_names=mesh_rules("singleton", "pos_embed_seq", "pos_embed_hidden"))
        vision_n_positions = n_patches + 1 if self.pooling_type == "CLS" else n_patches
        self.vision_position_ids = nnx.Param(jnp.arange(vision_n_positions, dtype=dtype).reshape(1, -1), sharding_names=mesh_rules("singleton", "pos_embed_seq"))

        if self.use_pre_norm:
            self.ln_pre = nnx.LayerNorm(
                hidden_size,
                epsilon=layernorm_epsilon,
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
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

        self.encoder = Transformer(
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            use_quick_gelu=use_quick_gelu,
            use_gradient_checkpointing=use_gradient_checkpointing,
            splash_attention_config=splash_attention_config,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            mesh=mesh,
            mesh_rules=mesh_rules,
        )

        self.ln_post = nnx.LayerNorm(
            hidden_size,
            epsilon=layernorm_epsilon,
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

    def __call__(self, img: Float[Array, "batch height width channels"]) -> Float[Array, "batch hidden_size"]:
        """
        Apply the Vision Transformer to input images.

        Args:
            img: Float[Array, "batch height width channels"]
                Batch of input images.

        Returns:
            Float[Array, "batch hidden_size"]
                Batch of output embeddings from the pooling method ([CLS] token or MultiheadAttentionPooling Head).
        """
        patches: Float[Array, "batch patches_h patches_w hidden_size"] = self.patch_embeddings(img)
        batch_size = patches.shape[0]
        patches: Float[Array, "batch n_patches hidden_size"] = patches.reshape(batch_size, -1, patches.shape[-1])
        if self.pooling_type == "CLS":
            cls_token: Float[Array, "batch 1 hidden_size"] = jnp.tile(self.cls_token[...], [batch_size, 1, 1])
            x: Float[Array, "batch n_patches+1 hidden_size"] = jnp.concat([cls_token, patches], axis=1)
        else:
            x: Float[Array, "batch n_patches hidden_size"] = patches
        embeddings: Float[Array, "batch length hidden_size"] = x + self.position_embeddings[...]  # length is either n_patches or n_patches+1 based on pooling type

        if self.use_pre_norm:
            x: Float[Array, "batch length hidden_size"] = self.ln_pre(embeddings)
        else:
            x: Float[Array, "batch length hidden_size"] = self.dropout(embeddings)

        x: Float[Array, "batch length hidden_size"] = self.encoder(x)
        x: Float[Array, "batch length hidden_size"] = self.ln_post(x)
        if self.pooling_type == "CLS":
            return x[:, 0]
        else:
            return self.MAPHead(x)
