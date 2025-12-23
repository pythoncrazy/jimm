import os
import time
from typing import Dict, Tuple

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
from flax import nnx
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int
from transformers import CLIPTokenizer

from jimm.models import CLIP  # noqa

tf.config.set_visible_devices([], "GPU")

PER_DEVICE_BATCH_SIZE = 2
GLOBAL_BATCH_SIZE = PER_DEVICE_BATCH_SIZE * jax.device_count()
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4
MAX_SEQ_LENGTH = 77
IMAGE_SIZE = 336
HF_MODEL_NAME = "geolocal/StreetCLIP"
mesh = Mesh(jax.devices(), ("fsdp",))
jax.set_mesh(mesh)
if jax.process_index() == 0:
    print(mesh)
    print(jax.devices())
    print(jax.local_devices())


def preprocess_text(texts: list[str], tokenizer: CLIPTokenizer, max_length: int = MAX_SEQ_LENGTH) -> Int[np.ndarray, "batch seq_len"]:
    """Tokenize and pad text strings.

    Args:
        texts (list[str]): List of text strings
        tokenizer (AutoTokenizer): HuggingFace tokenizer
        max_length (int): Maximum sequence length

    Returns:
        Int[Array, "batch seq_len"]: Tokenized and padded text
    """
    encoded: np.ndarray = tokenizer(texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="np")
    return encoded["input_ids"].astype("int32")


def preprocess_images(images: np.ndarray) -> Float[np.ndarray, "batch height width channels"]:
    """Preprocess images to [-1, 1] range.

    Args:
        images (Array): Raw image array

    Returns:
        Float[Array, "batch height width channels"]: Normalized images
    """
    if images.shape[-1] != 3:
        images: np.ndarray = np.transpose(images, (0, 2, 3, 1))
    return (images.astype("float32") / 255.0) * 2.0 - 1.0


def clip_loss_fn(image_features: Float[Array, "batch embed_dim"], text_features: Float[Array, "batch embed_dim"], logit_scale: Float[Array, ""]) -> Float[Array, ""]:
    """Compute CLIP contrastive loss.

    Args:
        image_features (Float[Array, "batch embed_dim"]): Image features
        text_features (Float[Array, "batch embed_dim"]): Text features
        logit_scale (Float[Array, ""]): Learnable temperature parameter

    Returns:
        Float[Array, ""]: Contrastive loss
    """
    image_features = image_features / jnp.linalg.norm(image_features, axis=-1, keepdims=True)
    text_features = text_features / jnp.linalg.norm(text_features, axis=-1, keepdims=True)
    logits = jnp.exp(logit_scale) * image_features @ text_features.T
    labels = jnp.arange(image_features.shape[0])
    image_loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    text_loss = optax.softmax_cross_entropy_with_integer_labels(logits.T, labels)
    return (image_loss.mean() + text_loss.mean()) / 2.0


def compute_loss_and_metrics(model: CLIP, images: Float[Array, "batch height width channels"], texts: Int[Array, "batch seq_len"]) -> Tuple[Float[Array, ""], Dict[str, Float[Array, ""]]]:
    """Compute loss and accuracy metrics.

    Args:
        model (CLIP): CLIP model
        images (Float[Array, "batch height width channels"]): Batch of images
        texts (Int[Array, "batch seq_len"]): Batch of text tokens

    Returns:
        Tuple[Float[Array, ""], Dict[str, Float[Array, ""]]]: Loss and metrics dictionary
    """
    image_features = model.encode_image(images, do_projection=True)
    text_features = model.encode_text(texts)
    loss = clip_loss_fn(image_features, text_features, model.logit_scale[...])
    image_features_norm = image_features / jnp.linalg.norm(image_features, axis=-1, keepdims=True)
    text_features_norm = text_features / jnp.linalg.norm(text_features, axis=-1, keepdims=True)
    logits = jnp.exp(model.logit_scale[...]) * image_features_norm @ text_features_norm.T
    predictions = jnp.argmax(logits, axis=-1)
    labels = jnp.arange(images.shape[0])
    accuracy = jnp.mean(predictions == labels)
    return loss, {"accuracy": accuracy, "logit_scale": model.logit_scale[...]}


def train_step_impl(
    model: CLIP, optimizer: nnx.Optimizer, images: Float[Array, "batch height width channels"], texts: Int[Array, "batch seq_len"]
) -> Tuple[Float[Array, ""], Dict[str, Float[Array, ""]]]:
    """Training step implementation.

    Args:
        model (CLIP): CLIP model
        optimizer (nnx.Optimizer): NNX optimizer
        images (Float[Array, "batch height width channels"]): Batch of images
        texts (Int[Array, "batch seq_len"]): Batch of text tokens

    Returns:
        Tuple[Float[Array, ""], Dict[str, Float[Array, ""]]]: Loss and metrics
    """

    def loss_fn(model):
        return compute_loss_and_metrics(model, images, texts)

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(model)
    optimizer.update(model, grads)
    return loss, metrics


def create_synthetic_dataset(num_samples: int = 1000) -> tf.data.Dataset:
    """Create synthetic image-text dataset.

    Args:
        num_samples (int): Number of samples to generate

    Returns:
        tf.data.Dataset: Synthetic dataset
    """
    captions = ["a photo of a cat", "a photo of a dog", "a picture of a bird", "an image of a car", "a photo of a tree", "a picture of a house", "an image of a person", "a photo of food"]

    def generate_sample(_) -> Dict[str, tf.Tensor]:
        image = tf.random.uniform([IMAGE_SIZE, IMAGE_SIZE, 3], 0, 255, dtype=tf.float32, seed=42)
        image = tf.cast(image, tf.uint8)
        caption_idx = tf.random.uniform([], 0, len(captions), dtype=tf.int32, seed=42)
        text = tf.gather(captions, caption_idx)
        return {"image": image, "text": text}

    dataset = tf.data.Dataset.range(num_samples)
    dataset = dataset.map(generate_sample)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def host_local_to_global_arrays(local_images: np.ndarray, local_texts: np.ndarray, mesh: Mesh) -> Tuple[Float[Array, "global_batch height width channels"], Int[Array, "global_batch seq_len"]]:
    """Convert host-local numpy arrays to globally sharded JAX arrays.

    Args:
        local_images (np.ndarray): Host-local image batch of shape (local_batch, height, width, channels)
        local_texts (np.ndarray): Host-local text batch of shape (local_batch, seq_len)
        mesh (Mesh): JAX mesh defining the global sharding layout

    Returns:
        Tuple[Float[Array, "global_batch height width channels"], Int[Array, "global_batch seq_len"]]:
            Tuple containing globally sharded image arrays and globally sharded text arrays
    """
    image_pspec = P("fsdp", None, None, None)
    text_pspec = P("fsdp", None)
    global_images = multihost_utils.host_local_array_to_global_array(local_images, mesh, image_pspec)
    global_texts = multihost_utils.host_local_array_to_global_array(local_texts, mesh, text_pspec)
    return global_images, global_texts


def load_and_shard_batch(batch: Dict[str, np.ndarray], tokenizer: CLIPTokenizer, mesh: Mesh) -> Tuple[Float[Array, "batch height width channels"], Int[Array, "batch seq_len"]]:
    """Load and shard batch across devices.

    Args:
        batch (Dict[str, np.ndarray]): Numpy batch dictionary
        tokenizer (AutoTokenizer): Text tokenizer
        mesh (Mesh): Device mesh

    Returns:
        Tuple[Float[Array, "batch height width channels"], Int[Array, "batch seq_len"]]: Sharded JAX arrays
    """
    images = preprocess_images(batch["image"])
    texts = [text.decode("utf-8") for text in batch["text"]]
    text_tokens = preprocess_text(texts, tokenizer)
    return host_local_to_global_arrays(images, jnp.asarray(text_tokens, dtype=jnp.float32), mesh)


@nnx.jit
def create_sharded_model_and_optimizer() -> Tuple[CLIP, nnx.Optimizer]:
    """Create and shard the CLIP model and optimizer following FSDP pattern.

    Returns:
        Tuple[CLIP, nnx.Optimizer]: Sharded model and optimizer
    """
    model = CLIP.from_pretrained(HF_MODEL_NAME, use_pytorch=True, use_gradient_checkpointing=True, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=nnx.Rngs(0))
    state = nnx.state(model, nnx.Param)
    pspecs = nnx.get_named_sharding(state, mesh=mesh)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state)
    optimizer = nnx.Optimizer(model, optax.adam(LEARNING_RATE), wrt=nnx.Param)
    state = nnx.state(optimizer, nnx.Param)
    pspecs = nnx.get_named_sharding(state, mesh=mesh)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(optimizer, sharded_state)
    return model, optimizer


def create_sharded_dataset(ds_raw: tf.data.Dataset, global_batch_size: int) -> tf.data.Dataset:
    """Create per-process sharded TensorFlow dataset for distributed training.

    Args:
        ds_raw (tf.data.Dataset): Raw TensorFlow dataset to shard
        global_batch_size (int): Total batch size across all processes

    Returns:
        tf.data.Dataset: Sharded dataset with local batch size per process
    """
    num_processes = jax.process_count()
    process_index = jax.process_index()
    local_batch_size = global_batch_size // num_processes
    sharded_ds = ds_raw.shard(num_shards=num_processes, index=process_index)
    sharded_ds = sharded_ds.batch(local_batch_size, drop_remainder=True)
    sharded_ds = sharded_ds.prefetch(tf.data.AUTOTUNE)
    return sharded_ds


def main() -> None:
    """Main training function."""

    with mesh:
        model, optimizer = create_sharded_model_and_optimizer()

    jax.debug.visualize_array_sharding(model.vision_model.visual_projection.kernel.value)

    model_spec = nnx.StateSharding(nnx.get_named_sharding(nnx.state(model), mesh))
    optimizer_spec = nnx.StateSharding(nnx.get_named_sharding(nnx.state(optimizer), mesh))
    image_sharding = NamedSharding(mesh, P("fsdp", None, None, None))
    text_sharding = NamedSharding(mesh, P("fsdp", None))

    train_step = nnx.jit(train_step_impl, in_shardings=(model_spec, optimizer_spec, image_sharding, text_sharding), out_shardings=(NamedSharding(mesh, P()), NamedSharding(mesh, P())))

    tokenizer = CLIPTokenizer.from_pretrained(HF_MODEL_NAME)
    num_train_samples = GLOBAL_BATCH_SIZE * 2
    train_dataset_raw = create_synthetic_dataset(num_train_samples)
    train_dataset = create_sharded_dataset(train_dataset_raw.repeat(NUM_EPOCHS), GLOBAL_BATCH_SIZE)

    model.train()
    total_steps = (num_train_samples * NUM_EPOCHS) // GLOBAL_BATCH_SIZE
    train_iterator = train_dataset.as_numpy_iterator()
    if jax.process_index() == 0:
        jax.profiler.start_trace("./tmp/profile-data")
    for step, batch in enumerate(train_iterator):
        start_time = time.time()
        images, texts = load_and_shard_batch(batch, tokenizer, mesh)
        loss, metrics = train_step(model, optimizer, images, texts)

        step_time = time.time() - start_time
        print(f"Step {step + 1}/{total_steps}: Loss={loss}, Time={step_time}s")
    if jax.process_index() == 0:
        jax.profiler.stop_trace()


if __name__ == "__main__":
    main()
