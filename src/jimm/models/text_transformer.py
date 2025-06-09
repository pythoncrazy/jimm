from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Int

from ..common.transformer import TransformerEncoder, sharded_init


class TextTransformer(nnx.Module):
    """A basic Text Transformer model."""

    def __init__(
        self,
        vocab_size: int,
        max_len: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        hidden_size: int = 512,
        dropout_rate: float = 0.1,
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(0),
        mesh: Optional[Mesh] = None,
    ) -> None:
        """Initialize a Text Transformer.

        Args:
            vocab_size (int): Size of the vocabulary.
            max_len (int): Maximum sequence length for positional embeddings.
            num_layers (int): Number of transformer layers.
            num_heads (int): Number of attention heads.
            mlp_dim (int): Size of the MLP dimension in transformer layers.
            hidden_size (int): Size of the hidden dimension.
            dropout_rate (float): Dropout rate.
            dtype (DTypeLike): Data type for computations.
            param_dtype (DTypeLike): Data type for parameters.
            rngs (nnx.Rngs): Random number generator keys.
            mesh (Optional[Mesh]): Optional JAX device mesh for parameter sharding.
        """
        self.token_embeddings = nnx.Embed(
            num_embeddings=vocab_size,
            features=hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            embedding_init=sharded_init(jax.nn.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0), P(None, "model"), mesh),
        )

        pos_emb_initializer = jax.nn.initializers.truncated_normal(stddev=0.02)
        pos_emb_value_unsharded = pos_emb_initializer(rngs.params(), (1, max_len, hidden_size), dtype=dtype)
        if mesh is not None:
            pos_emb_value_sharded = jax.device_put(pos_emb_value_unsharded, NamedSharding(mesh, P(None, None, "model")))
            self.position_embeddings = nnx.Param(pos_emb_value_sharded)
        else:
            self.position_embeddings = nnx.Param(pos_emb_value_unsharded)

        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

        self.encoder = nnx.Sequential(
            *[
                TransformerEncoder(
                    hidden_size,
                    mlp_dim,
                    num_heads,
                    dropout_rate,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    rngs=rngs,
                    mesh=mesh,
                )
                for _ in range(num_layers)
            ]
        )
        # Optional: Add a final LayerNorm if desired, similar to ViT's final_norm
        # self.final_norm = nnx.LayerNorm(...)

        # Optional: Add a classification or language modeling head here
        # For example, a linear layer for classification:
        # self.classifier_head = nnx.Linear(hidden_size, num_classes, ...)

    def __call__(self, input_ids: Int[Array, "batch seq_len"]) -> Float[Array, "batch seq_len hidden_size"]:
        """Forward pass of the Text Transformer.

        Args:
            input_ids: Input token IDs with shape [batch, sequence_length].

        Returns:
            Output tensor with shape [batch, sequence_length, hidden_size].
        """
        seq_len = input_ids.shape[1]
        if seq_len > self.position_embeddings.value.shape[1]:
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds maximum "
                f"positional embedding length ({self.position_embeddings.value.shape[1]})"
            )

        token_embeds = self.token_embeddings(input_ids)
        # Slice positional embeddings to match input sequence length
        position_embeds = self.position_embeddings.value[:, :seq_len, :]

        embeddings = token_embeds + position_embeds
        embeddings = self.dropout(embeddings)

        encoded_output = self.encoder(embeddings)

        # If final_norm and/or a head is added, apply them here.
        # e.g., encoded_output = self.final_norm(encoded_output)
        # e.g., logits = self.classifier_head(encoded_output[:, 0]) # For CLS token style classification

        return encoded_output
