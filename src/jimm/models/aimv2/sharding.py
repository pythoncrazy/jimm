import dataclasses

from jimm.models.vit.sharding import ViTSharding


@dataclasses.dataclass(frozen=True)
class AIMv2Sharding(ViTSharding):
    """FSDP sharding for AIMv2, targeting up to 256 devices.

    AIMv2 has no LayerScale, register tokens, or MLP biases, so ViTSharding
    defaults are used without modification.
    """
