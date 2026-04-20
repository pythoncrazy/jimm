import dataclasses


@dataclasses.dataclass(frozen=True)
class SigLIPSharding:
    """FSDP sharding for SigLIP.

    Specs represent per-layer (non-stacked) shapes. Transformer.__init__
    prepends None for the scan axis to Variable metadata after vmap so the
    optimizer sees the correct stacked spec.

    Large matrices are sharded on the contracting (in_features) dimension so
    activations carry only batch-axis sharding.
    """

    attn_qkv_kernel: tuple[str | None, str | None, str | None] = ("fsdp", None, None)
    attn_qkv_bias: tuple[str | None, str | None] = (None, None)
    attn_out_kernel: tuple[str | None, str | None, str | None] = ("fsdp", None, None)
    attn_out_bias: tuple[str | None] = (None,)
    mlp_up_kernel: tuple[str | None, str | None] = ("fsdp", None)
    mlp_up_bias: tuple[str | None] = (None,)
    mlp_down_kernel: tuple[str | None, str | None] = ("fsdp", None)
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
    proj_kernel: tuple[str | None, str | None] = ("fsdp", None)
    proj_bias: tuple[str | None] = (None,)
