import dataclasses
from typing import Protocol


class ShardingSpec(Protocol):
    """Protocol defining the sharding specification for model parameters.

    Each attribute is a tuple specifying the sharding for each dimension of the tensor.
    The tuple length must match the tensor's number of dimensions.
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
    cls_token: tuple[str | None, str | None, str | None]
    probe_token: tuple[str | None, str | None, str | None]
    proj_kernel: tuple[str | None, str | None]
    proj_bias: tuple[str | None]


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
    cls_token: tuple[str | None, str | None, str | None] = (None, None, None)
    probe_token: tuple[str | None, str | None, str | None] = (None, None, None)
    proj_kernel: tuple[str | None, str | None] = (None, None)
    proj_bias: tuple[str | None] = (None,)
