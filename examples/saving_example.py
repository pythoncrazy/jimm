import jax
import os
from flax import nnx
from jimm.models import CLIP
from jax.experimental import mesh_utils
from jax.sharding import Mesh


HF_MODEL_NAME = "openai/clip-vit-large-patch14"
SAVE_DIR = "tmp/saved_clip_model"

devices = mesh_utils.create_device_mesh((1, jax.device_count()))
mesh = Mesh(devices, ("batch", "model"))


@nnx.jit
def create_sharded_model() -> CLIP:
    """Create and shard the CLIP model following FSDP pattern."""
    model = CLIP.from_pretrained(HF_MODEL_NAME, use_gradient_checkpointing=True, rngs=nnx.Rngs(0))
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state)
    return model


print("Loading and sharding model...")
with mesh:
    model = create_sharded_model()

print(f"Saving model to {SAVE_DIR}...")
model.save_pretrained(SAVE_DIR)

print("Verifying saved files:")
for file in os.listdir(SAVE_DIR):
    file_path = os.path.join(SAVE_DIR, file)
    size_mb = os.path.getsize(file_path) / 1024 / 1024
    print(f"  {file}: {size_mb:.2f} MB")

print("\nTesting reload from saved model...")
reloaded_model = CLIP.from_pretrained(SAVE_DIR)
print("✓ Successfully reloaded model from saved files")
