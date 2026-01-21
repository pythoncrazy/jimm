import jax
import jax.numpy as jnp
import qwix
from flax import nnx
from jax.sharding import Mesh

from jimm import CLIP

MAX_SEQ_LENGTH = 77
IMAGE_SIZE = 336
HF_MODEL_NAME = "geolocal/StreetCLIP"
mesh = Mesh(jax.devices(), ("fsdp",))
jax.set_mesh(mesh)
jax.profiler.start_trace("/tmp/profile-data")
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

model = CLIP.from_pretrained(HF_MODEL_NAME, use_pytorch=True, use_gradient_checkpointing=True, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=nnx.Rngs(0))
qt_rules = [
    qwix.QuantizationRule(
        module_path=".*",  # this rule matches all modules.
        weight_qtype=jnp.int8,  # does not quantizes weights in int8.
        act_qtype=jnp.int8,  # quantizes activations in int8.
    )
]
model_input = {
    "images": jax.random.uniform(jax.random.key(0), (1, IMAGE_SIZE, IMAGE_SIZE, 3)),
    "text": jnp.ones((1, MAX_SEQ_LENGTH), dtype=jnp.int32),
}
ptq_model = qwix.quantize_model(model, qwix.PtqProvider(qt_rules), image=model_input["images"], text=model_input["text"])
print(jax.eval_shape(nnx.to_pure_dict, nnx.state(ptq_model)))
jax.profiler.stop_trace()
