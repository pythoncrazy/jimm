import dataclasses


@dataclasses.dataclass(frozen=True)
class SigLIPSharding:
    """FSDP sharding for SigLIP.

    Specs represent per-layer (non-stacked) shapes. Transformer.__init__
    prepends None for the scan axis to Variable metadata after vmap so the
    optimizer sees the correct stacked spec.

    All models use num_heads = hidden // 64, so head_dim = 64 always.

        Model    hidden  heads  head_dim  mlp_dim   hidden÷256  mlp÷256
        Base       768     12       64     3072          3        12
        Large     1024     16       64     4096          4        16
        So400m    1152     18       64     4608          X        18
        H         1280     20       64     5120          5        20
        G         1664     26       64     8192          X        32

    Sharding choices:
        attn_qkv_kernel (hidden, heads, head_dim): shard ax0 (hidden, contracting).
            Divides by 256 for Base/Large/H; by 128 for all models.
        attn_out_kernel (heads, head_dim, hidden): NOT sharded.
            ax2 (hidden) is a free output dim — produces [batch@fsdp, seq, hidden@fsdp]
            when batch is also on fsdp (illegal double-sharding).
            ax1 (head_dim=64) caps at 64 devices. ax0 (heads) never divides large FSDP.
        mlp_up_kernel (hidden, mlp_dim): shard ax0 (hidden, contracting).
            Same divisibility as qkv_kernel.
        mlp_down_kernel (mlp_dim, hidden): shard ax0 (mlp_dim, contracting).
            mlp_dim divides by 512 for all models.
        embed: shard ax0 (vocab_size, SigLIP vocab 32000 ÷ 256 = 125 ✓).
        pos_embed_3d: shard ax2 (hidden_size).
        patch_conv_kernel: shard ax3 (out_channels = hidden_size).
    """

    attn_qkv_kernel: tuple[str | None, str | None, str | None] = ("fsdp", None, None)
    attn_qkv_bias: tuple[str | None, str | None] = (None, None)
    attn_out_kernel: tuple[str | None, str | None, str | None] = (None, None, None)
    attn_out_bias: tuple[str | None] = (None,)
    mlp_up_kernel: tuple[str | None, str | None] = ("fsdp", None)
    mlp_up_bias: tuple[str | None] = (None,)
    mlp_down_kernel: tuple[str | None, str | None] = ("fsdp", None)
    mlp_down_bias: tuple[str | None] = (None,)
    layernorm: tuple[str | None] = (None,)
    patch_conv_kernel: tuple[str | None, str | None, str | None, str | None] = (None, None, None, "fsdp")
    patch_conv_bias: tuple[str | None] = (None,)
    embed: tuple[str | None, str | None] = ("fsdp", None)
    pos_embed_3d: tuple[str | None, str | None, str | None] = (None, None, "fsdp")
    pos_embed_2d: tuple[str | None, str | None] = (None, None)
    vision_pos_id: tuple[str | None, str | None] = (None, None)
    text_pos_embed: tuple[str | None, str | None] = (None, None)
    cls_token: tuple[str | None, str | None, str | None] = (None, None, None)
    probe_token: tuple[str | None, str | None, str | None] = (None, None, None)
    proj_kernel: tuple[str | None, str | None] = ("fsdp", None)
    proj_bias: tuple[str | None] = (None,)
