from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
from flax import nnx
from jax.experimental import mesh_utils
from jax.sharding import Mesh, set_mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int
from transformers import AutoTokenizer

jax.distributed.initialize()


from jimm.common.utils import get_fsdp_sharding_specs  # noqa
from jimm.models.clip import CLIP  # noqa

tf.config.set_visible_devices([], "GPU")


GLOBAL_BATCH_SIZE = 512
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4
MAX_SEQ_LENGTH = 77
IMAGE_SIZE = 336

HF_MODEL_NAME = "geolocal/StreetCLIP"

mesh = None


def visualize_array_sharding(array: Array, name: str) -> None:
    """Visualize sharding of arrays with proper handling for different dimensions.

    Args:
        array: JAX array to visualize
        name: Name identifier for the array
    """
    if jax.process_index() == 0:
        print(f"{name}: JAX type = {jax.typeof(array)}")
        print(f"{name}: Concrete sharding = {array.sharding}")
        if array.ndim == 0:
            print(f"{name} (scalar): {array.sharding}")
        elif array.ndim == 1:
            print(f"{name} (1D, shape {array.shape}): {array.sharding}")
            jax.debug.visualize_array_sharding(array)
        elif array.ndim == 2:
            print(f"{name} (2D, shape {array.shape}): {array.sharding}")
            jax.debug.visualize_array_sharding(array)
        elif array.ndim >= 3:
            print(f"{name} ({array.ndim}D, shape {array.shape}): {array.sharding}")
        print()


def visualize_model_sharding(model: nnx.Module) -> None:
    """Visualize sharding of all model parameters.

    Args:
        model: Flax NNX model to visualize
    """
    print("=== Model Parameter Sharding ===")
    state = nnx.state(model)
    flat_state = nnx.to_flat_state(state)

    for key, param in flat_state:
        if hasattr(param, "value"):
            param_name = ".".join(str(k) for k in key)
            visualize_array_sharding(param.value, param_name)
            print()


def visualize_optimizer_sharding(optimizer: nnx.Optimizer) -> None:
    """Visualize sharding of optimizer state.

    Args:
        optimizer: Flax NNX optimizer to visualize
    """
    print("=== Optimizer State Sharding ===")
    state = nnx.state(optimizer)
    flat_state = nnx.to_flat_state(state)

    for key, param in flat_state:
        if hasattr(param, "value"):
            param_name = ".".join(str(k) for k in key)
            visualize_array_sharding(param.value, param_name)
            print()


mesh = None


def preprocess_text(texts: list[str], tokenizer: AutoTokenizer, max_length: int = MAX_SEQ_LENGTH) -> np.ndarray:
    """Tokenize and pad text strings.

    Args:
        texts: List of text strings
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length

    Returns:
        numpy array: Tokenized and padded text
    """
    encoded = tokenizer(texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="np")
    return encoded["input_ids"].astype("int32")


def preprocess_images(images: np.ndarray) -> np.ndarray:
    """Preprocess images to [-1, 1] range.

    Args:
        images: Raw image array

    Returns:
        numpy array: Normalized images
    """
    if images.shape[-1] != 3:
        images = np.transpose(images, (0, 2, 3, 1))
    return (images.astype("float32") / 255.0) * 2.0 - 1.0


def clip_loss_fn(image_features: Float[Array, "batch embed_dim"], text_features: Float[Array, "batch embed_dim"], logit_scale: Float[Array, ""]) -> Float[Array, ""]:
    """Compute CLIP contrastive loss.

    Args:
        image_features: Image features
        text_features: Text features
        logit_scale: Learnable temperature parameter

    Returns:
        Float[Array, ""]: Contrastive loss
    """
    image_features = image_features / jnp.linalg.norm(image_features, axis=-1, keepdims=True)
    text_features = text_features / jnp.linalg.norm(text_features, axis=-1, keepdims=True)

    logits = jnp.exp(logit_scale) * image_features @ text_features.T
    labels = jnp.arange(image_features.shape[0])

    image_loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    text_loss = optax.softmax_cross_entropy_with_integer_labels(logits.T, labels)

    loss = (image_loss.mean() + text_loss.mean()) / 2.0
    return loss


def compute_loss_and_metrics(model: CLIP, images: Float[Array, "batch height width channels"], texts: Int[Array, "batch seq_len"]) -> Float[Array, ""]:
    """Compute loss and accuracy metrics.

    Args:
        model: CLIP model
        images: Batch of images
        texts: Batch of text tokens

    Returns:
        Tuple of loss and metrics dictionary
    """
    image_features = model.encode_image(images)
    text_features = model.encode_text(texts)

    loss = clip_loss_fn(image_features, text_features, model.logit_scale.value)

    return loss


def train_step(model: CLIP, optimizer: nnx.Optimizer, images: Float[Array, "batch height width channels"], texts: Int[Array, "batch seq_len"]) -> Float[Array, ""]:
    """Training step implementation.

    Args:
        model: CLIP model
        optimizer: NNX optimizer
        images: Batch of images
        texts: Batch of text tokens

    Returns:
        Loss value
    """
    grad_fn = nnx.value_and_grad(compute_loss_and_metrics)
    loss, grads = grad_fn(model, images, texts)
    optimizer.update(grads)
    return loss


def create_synthetic_dataset(num_samples: int = 1000) -> tf.data.Dataset:
    """Create synthetic image-text dataset with fixed samples.

    Args:
        num_samples: Number of samples to generate

    Returns:
        tf.data.Dataset: Synthetic dataset with consistent samples
    """
    captions = [
        "a photo of a cat",
        "a photo of a dog",
        "a picture of a bird",
        "an image of a car",
        "a photo of a tree",
        "a picture of a house",
        "an image of a person",
        "a photo of food",
    ]

    fixed_image = tf.random.uniform([IMAGE_SIZE, IMAGE_SIZE, 3], 0, 255, seed=1337, dtype=tf.float32)
    fixed_image = tf.cast(fixed_image, tf.uint8)
    fixed_caption = captions[0]

    def generate_sample(_):
        return {"image": fixed_image, "text": fixed_caption}

    dataset = tf.data.Dataset.range(num_samples)
    dataset = dataset.map(generate_sample)
    return dataset


def preprocess_batch(batch: Dict[str, tf.Tensor], tokenizer: AutoTokenizer) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Preprocess batch data to jax arrays."""
    images = preprocess_images(batch["image"].numpy())
    texts = [text.decode("utf-8") for text in batch["text"].numpy()]
    text_tokens = preprocess_text(texts, tokenizer)
    return jnp.array(images), jnp.array(text_tokens)


def create_sharded_model_and_optimizer():
    """Create and shard the CLIP model and optimizer using nnx.with_sharding_constraint."""
    model = CLIP.from_pretrained(HF_MODEL_NAME, use_pytorch=True, rngs=nnx.Rngs(0))
    state = nnx.state(model)
    model_pspecs = get_fsdp_sharding_specs(state, mesh, fsdp_axis_name="fsdp", min_size_to_shard_mb=0)
    sharded_state = nnx.with_sharding_constraint(state, model_pspecs, mesh)
    nnx.update(model, sharded_state)

    optimizer = nnx.Optimizer(model, optax.adam(LEARNING_RATE))
    state = nnx.state(optimizer, nnx.optimizer.OptState)
    optimizer_pspecs = get_fsdp_sharding_specs(state, mesh, fsdp_axis_name="fsdp", min_size_to_shard_mb=0)
    sharded_state = nnx.with_sharding_constraint(state, optimizer_pspecs, mesh)
    nnx.update(optimizer, sharded_state)

    return model, optimizer


def main() -> None:
    """Main training function."""
    global mesh
    devices = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(devices, ("fsdp",))

    set_mesh(mesh)

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    train_dataset = create_synthetic_dataset(4096)
    train_dataset = train_dataset.batch(GLOBAL_BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    model, optimizer = create_sharded_model_and_optimizer()
    model_pspecs = nnx.StateSharding(get_fsdp_sharding_specs(nnx.state(model), mesh, fsdp_axis_name="fsdp", min_size_to_shard_mb=0))
    optimizer_pspecs = nnx.StateSharding(get_fsdp_sharding_specs(nnx.state(optimizer), mesh, fsdp_axis_name="fsdp", min_size_to_shard_mb=0))
    train_step_fsdp = nnx.jit(train_step, in_shardings=(model_pspecs, optimizer_pspecs, P("fsdp", None, None, None), P("fsdp", None)))

    visualize_model_sharding(model)
    visualize_optimizer_sharding(optimizer)

    for epoch in range(NUM_EPOCHS):
        model.train()
        losses = []

        for step, batch in enumerate(train_dataset.take(100)):
            images, texts = preprocess_batch(batch, tokenizer)
            images = jax.device_put(images, NamedSharding(mesh, P("fsdp", None, None, None)))
            texts = jax.device_put(texts, NamedSharding(mesh, P("fsdp", None)))
            if step == 0 and epoch == 0 and jax.process_index() == 0:
                print("Data sharding visualization:")
                visualize_array_sharding(images, "batch_images")
                visualize_array_sharding(texts, "batch_texts")

            loss = train_step_fsdp(model, optimizer, images, texts)
            losses.append(float(loss))
            print(f"step: {step}, loss: {loss}")


if __name__ == "__main__":
    main()
