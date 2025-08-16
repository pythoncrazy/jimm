import jax
from flax import nnx
from jax.experimental import mesh_utils
from jax.sharding import Mesh

from jimm.models.siglip import SigLIP, SigLIPVisionModel

HF_MODEL_NAME = "google/siglip-base-patch16-512"
SAVE_DIR = "tmp/saved_siglip_model"

devices = mesh_utils.create_device_mesh((1, jax.device_count()))
mesh = Mesh(devices, ("batch", "model"))


@nnx.jit
def create_sharded_model() -> SigLIP:
    """Create and shard the CLIP model following FSDP pattern."""
    model = SigLIP.from_pretrained(HF_MODEL_NAME, use_gradient_checkpointing=True, rngs=nnx.Rngs(0))
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state)
    return model


with mesh:
    model = create_sharded_model()

model.save_pretrained(SAVE_DIR)


reloaded_model = SigLIPVisionModel.from_pretrained(SAVE_DIR)
