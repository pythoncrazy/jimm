import dataclasses
from typing import Any, Protocol

import jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


class ShardingSpec(Protocol):
    """Protocol defining the sharding specification for model parameters.

    Specs represent per-layer (non-stacked) shapes. The Transformer stacks
    layers via nnx.vmap and patches the stacked Variable metadata to prepend
    None for the scan axis, so the optimizer sees the correct 4-D spec.

        attn_qkv_kernel  (in_features, num_heads, head_dim)
        attn_qkv_bias    (num_heads, head_dim)
        attn_out_kernel  (num_heads, head_dim, out_features)
        attn_out_bias    (out_features,)
        mlp_up_kernel    (in_features, intermediate_size)
        mlp_up_bias      (intermediate_size,)
        mlp_down_kernel  (intermediate_size, out_features)
        mlp_down_bias    (out_features,)
        layernorm        (hidden_size,)
    """

    attn_qkv_kernel: tuple[str | None, str | None, str | None]
    attn_qkv_bias: tuple[str | None, str | None]
    attn_out_kernel: tuple[str | None, str | None, str | None]
    attn_out_bias: tuple[str | None]
    mlp_up_kernel: tuple[str | None, str | None]
    mlp_up_bias: tuple[str | None]
    mlp_down_kernel: tuple[str | None, str | None]
    mlp_down_bias: tuple[str | None]
    layernorm: tuple[str | None]
    patch_conv_kernel: tuple[str | None, str | None, str | None, str | None]
    patch_conv_bias: tuple[str | None]
    embed: tuple[str | None, str | None]
    pos_embed_3d: tuple[str | None, str | None, str | None]
    pos_embed_2d: tuple[str | None, str | None]
    vision_pos_id: tuple[str | None, str | None]
    text_pos_embed: tuple[str | None, str | None]
    cls_token: tuple[str | None, str | None, str | None]
    probe_token: tuple[str | None, str | None, str | None]
    proj_kernel: tuple[str | None, str | None]
    proj_bias: tuple[str | None]
    layer_scale: tuple[str | None]


@dataclasses.dataclass(frozen=True)
class NoSharding:
    """No sharding - all parameters replicated."""

    attn_qkv_kernel: tuple[str | None, str | None, str | None] = (None, None, None)
    attn_qkv_bias: tuple[str | None, str | None] = (None, None)
    attn_out_kernel: tuple[str | None, str | None, str | None] = (None, None, None)
    attn_out_bias: tuple[str | None] = (None,)
    mlp_up_kernel: tuple[str | None, str | None] = (None, None)
    mlp_up_bias: tuple[str | None] = (None,)
    mlp_down_kernel: tuple[str | None, str | None] = (None, None)
    mlp_down_bias: tuple[str | None] = (None,)
    layernorm: tuple[str | None] = (None,)
    patch_conv_kernel: tuple[str | None, str | None, str | None, str | None] = (None, None, None, None)
    patch_conv_bias: tuple[str | None] = (None,)
    embed: tuple[str | None, str | None] = (None, None)
    pos_embed_3d: tuple[str | None, str | None, str | None] = (None, None, None)
    pos_embed_2d: tuple[str | None, str | None] = (None, None)
    vision_pos_id: tuple[str | None, str | None] = (None, None)
    text_pos_embed: tuple[str | None, str | None] = (None, None)
    cls_token: tuple[str | None, str | None, str | None] = (None, None, None)
    probe_token: tuple[str | None, str | None, str | None] = (None, None, None)
    proj_kernel: tuple[str | None, str | None] = (None, None)
    proj_bias: tuple[str | None] = (None,)
    layer_scale: tuple[str | None] = (None,)


def sharding_of(value: Any) -> NamedSharding:
    """Returns the traced NamedSharding for a value in explicit mode."""

    return jax.typeof(value).sharding


def named_sharding_like(reference: Any, spec: P | tuple[str | None, ...]) -> NamedSharding:
    """Builds a NamedSharding on the same mesh as the reference value."""

    if not isinstance(spec, P):
        spec = P(*spec)
    return NamedSharding(sharding_of(reference).mesh, spec)


def replicated_sharding(reference: Any) -> NamedSharding:
    """Builds a replicated sharding matching the reference mesh and rank."""

    return named_sharding_like(reference, P(*([None] * reference.ndim)))


def reshard_like(value: Any, reference: Any) -> Any:
    """Reshards a value to match the sharding of a reference value."""

    return jax.sharding.reshard(value, sharding_of(reference))
