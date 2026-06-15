import dataclasses

from jimm.models.vit.sharding import ViTSharding


@dataclasses.dataclass(frozen=True)
class DINOv3Sharding(ViTSharding):
    """FSDP sharding for DINOv3, targeting up to 256 devices.

    Extends ViTSharding with layer_scale sharding along fsdp.

    layer_scale vectors (shape: hidden_size) and gated-MLP biases are sharded
    along fsdp, matching the contracting axis of the corresponding weight matrices.
    """

    layer_scale: tuple[str | None] = ("fsdp",)
    mlp_up_bias: tuple[str | None] = ("fsdp",)
    register_tokens: tuple[str | None, str | None, str | None] = (None, None, None)
