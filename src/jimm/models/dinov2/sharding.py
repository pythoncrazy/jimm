import dataclasses

from jimm.models.vit.sharding import ViTSharding


@dataclasses.dataclass(frozen=True)
class Dinov2Sharding(ViTSharding):
    """FSDP sharding for DINOv2, targeting up to 256 devices.

    Extends ViTSharding with layer_scale sharding along fsdp.

    DINOv2-small has hidden_size=384 (not divisible by 256; JAX pads).
    DINOv2-base has hidden_size=768 (divides by 256 → 3 per shard).

    layer_scale vectors (shape: hidden_size) are sharded along fsdp,
    matching the same axis as attn_qkv and mlp weight contracting dims.
    """

    mlp_up_bias: tuple[str | None] = ("fsdp",)
    layer_scale: tuple[str | None] = ("fsdp",)
