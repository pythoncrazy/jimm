import dataclasses


@dataclasses.dataclass(frozen=True)
class ViTSharding:
    """FSDP sharding for ViT.

    Specs represent per-layer (non-stacked) shapes. Transformer.__init__
    prepends None for the scan axis to Variable metadata after vmap so the
    optimizer sees the correct stacked spec.

    attn_qkv_kernel shards on in_features (hidden_size, divisible by 256 for all supported models).
    attn_out_kernel shards on out_features (axis 2 = hidden_size) — num_heads (axis 0 ≤ 16) cannot
    divide 64+ FSDP devices.
    mlp_up_kernel shards on in_features (hidden_size, contracting axis — keeps activations unsharded).
    mlp_down_kernel shards on intermediate_size (4*hidden_size, contracting axis, axis 0).
    attn_out_bias shards on out_features consistent with attn_out_kernel.
    pos_embed_3d shards on hidden_size (axis 2, divisible by 256 for all supported models).
    patch_conv_kernel shards on out_channels (axis 3 = hidden_size).
    """

    attn_qkv_kernel: tuple[str | None, str | None, str | None] = ("fsdp", None, None)
    attn_qkv_bias: tuple[str | None, str | None] = (None, None)
    attn_out_kernel: tuple[str | None, str | None, str | None] = (None, None, "fsdp")
    attn_out_bias: tuple[str | None] = ("fsdp",)
    mlp_up_kernel: tuple[str | None, str | None] = ("fsdp", None)
    mlp_up_bias: tuple[str | None] = (None,)
    mlp_down_kernel: tuple[str | None, str | None] = ("fsdp", None)
    mlp_down_bias: tuple[str | None] = (None,)
    layernorm: tuple[str | None] = (None,)
    patch_conv_kernel: tuple[str | None, str | None, str | None, str | None] = (None, None, None, "fsdp")
    patch_conv_bias: tuple[str | None] = (None,)
    embed: tuple[str | None, str | None] = (None, None)
    pos_embed_3d: tuple[str | None, str | None, str | None] = (None, None, "fsdp")
    pos_embed_2d: tuple[str | None, str | None] = (None, None)
    vision_pos_id: tuple[str | None, str | None] = (None, None)
    text_pos_embed: tuple[str | None, str | None] = (None, None)
    cls_token: tuple[str | None, str | None, str | None] = (None, None, None)
    probe_token: tuple[str | None, str | None, str | None] = (None, None, None)
    proj_kernel: tuple[str | None, str | None] = ("fsdp", None)
    proj_bias: tuple[str | None] = (None,)
