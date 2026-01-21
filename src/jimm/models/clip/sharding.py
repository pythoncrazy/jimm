import dataclasses


@dataclasses.dataclass(frozen=True)
class CLIPSharding:
    """FSDP sharding for CLIP."""

    attn_qkv_kernel: tuple[str | None, str | None, str | None] = ("fsdp", None, None)
    attn_qkv_bias: tuple[str | None, str | None] = (None, None)
    attn_out_kernel: tuple[str | None, str | None, str | None] = (None, None, "fsdp")
    attn_out_bias: tuple[str | None] = ("fsdp",)
    mlp_up_kernel: tuple[str | None, str | None] = ("fsdp", None)
    mlp_up_bias: tuple[str | None] = (None,)
    mlp_down_kernel: tuple[str | None, str | None] = (None, "fsdp")
    mlp_down_bias: tuple[str | None] = ("fsdp",)
    layernorm: tuple[str | None] = ("fsdp",)
    patch_conv_kernel: tuple[str | None, str | None, str | None, str | None] = (None, None, None, "fsdp")
    patch_conv_bias: tuple[str | None] = ("fsdp",)
    embed: tuple[str | None, str | None] = ("fsdp", None)
    pos_embed_3d: tuple[str | None, str | None, str | None] = (None, None, "fsdp")
    pos_embed_2d: tuple[str | None, str | None] = (None, None)
    vision_pos_id: tuple[str | None, str | None] = (None, None)
    text_pos_embed: tuple[str | None, str | None] = (None, None)
    cls_token: tuple[str | None, str | None, str | None] = (None, None, "fsdp")
    probe_token: tuple[str | None, str | None, str | None] = (None, None, "fsdp")
    proj_kernel: tuple[str | None, str | None] = ("fsdp", None)
    proj_bias: tuple[str | None] = (None,)
