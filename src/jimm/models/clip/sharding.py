import dataclasses


@dataclasses.dataclass(frozen=True)
class CLIPSharding:
    """FSDP sharding for CLIP.

    Transformer layer params have a leading num_layers dimension from nnx.scan,
    so each scanned-param spec has a leading None for that dimension.

    All large matrices are sharded on the contracting (in_features) dimension so
    that activations only carry batch-axis sharding, avoiding duplicate-axis
    conflicts in explicit sharding mode.
    """

    attn_qkv_kernel: tuple[str | None, str | None, str | None, str | None] = (None, "fsdp", None, None)
    attn_qkv_bias: tuple[str | None, str | None, str | None] = (None, None, None)
    attn_out_kernel: tuple[str | None, str | None, str | None, str | None] = (None, "fsdp", None, None)
    attn_out_bias: tuple[str | None, str | None] = (None, None)
    mlp_up_kernel: tuple[str | None, str | None, str | None] = (None, "fsdp", None)
    mlp_up_bias: tuple[str | None, str | None] = (None, None)
    mlp_down_kernel: tuple[str | None, str | None, str | None] = (None, "fsdp", None)
    mlp_down_bias: tuple[str | None, str | None] = (None, None)
    layernorm: tuple[str | None, str | None] = (None, None)
    patch_conv_kernel: tuple[str | None, str | None, str | None, str | None] = (None, None, None, None)
    patch_conv_bias: tuple[str | None] = (None,)
    embed: tuple[str | None, str | None] = ("fsdp", None)
    pos_embed_3d: tuple[str | None, str | None, str | None] = (None, None, None)
    pos_embed_2d: tuple[str | None, str | None] = (None, None)
    vision_pos_id: tuple[str | None, str | None] = (None, None)
    text_pos_embed: tuple[str | None, str | None] = (None, None)
    cls_token: tuple[str | None, str | None, str | None] = (None, None, None)
    probe_token: tuple[str | None, str | None, str | None] = (None, None, None)
    proj_kernel: tuple[str | None, str | None] = ("fsdp", None)
    proj_bias: tuple[str | None] = (None,)
