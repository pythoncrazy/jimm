import dataclasses

from jimm.models.vit.sharding import ViTSharding


@dataclasses.dataclass(frozen=True)
class AIMv2Sharding(ViTSharding):
    """FSDP sharding for AIMv2.

    AIMv2 has no LayerScale, register tokens, or MLP biases, so ViTSharding
    defaults are used without modification.
    """
