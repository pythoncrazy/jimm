from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
from flax import nnx
from jax.experimental import mesh_utils
from jax.sharding import Mesh
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


def compute_loss_and_metrics(model: CLIP, images: Float[Array, "batch height width channels"], texts: Int[Array, "batch seq_len"]) -> Tuple[Float[Array, ""], Dict[str, Float[Array, ""]]]:
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

    image_features_norm = image_features / jnp.linalg.norm(image_features, axis=-1, keepdims=True)
    text_features_norm = text_features / jnp.linalg.norm(text_features, axis=-1, keepdims=True)
    logits = jnp.exp(model.logit_scale.value) * image_features_norm @ text_features_norm.T
    predictions = jnp.argmax(logits, axis=-1)
    labels = jnp.arange(images.shape[0])
    accuracy = jnp.mean(predictions == labels)

    return loss, {"accuracy": accuracy, "logit_scale": model.logit_scale.value}


def train_fn(model: CLIP, optimizer: nnx.Optimizer, images: Float[Array, "batch height width channels"], texts: Int[Array, "batch seq_len"]) -> Tuple[Float[Array, ""], Dict[str, Float[Array, ""]]]:
    """Training step implementation.

    Args:
        model: CLIP model
        optimizer: NNX optimizer
        images: Batch of images
        texts: Batch of text tokens

    Returns:
        Tuple of loss and metrics
    """
    grad_fn = nnx.value_and_grad(compute_loss_and_metrics, has_aux=True)
    (loss, metrics), grads = grad_fn(model, images, texts)
    optimizer.update(grads)
    return loss, metrics


def create_synthetic_dataset(num_samples: int = 1000) -> tf.data.Dataset:
    """Create synthetic image-text dataset.

    Args:
        num_samples: Number of samples to generate

    Returns:
        tf.data.Dataset: Synthetic dataset
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

    def generate_sample(_):
        image = tf.random.uniform([IMAGE_SIZE, IMAGE_SIZE, 3], 0, 255, dtype=tf.float32)
        image = tf.cast(image, tf.uint8)
        caption_idx = tf.random.uniform([], 0, len(captions), dtype=tf.int32)
        text = tf.gather(captions, caption_idx)
        return {"image": image, "text": text}

    dataset = tf.data.Dataset.range(num_samples)
    dataset = dataset.map(generate_sample)
    return dataset


def preprocess_batch(batch: Dict[str, tf.Tensor], tokenizer: AutoTokenizer) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Preprocess batch data to jax arrays."""
    images = preprocess_images(batch["image"].numpy())
    texts = [text.decode("utf-8") for text in batch["text"].numpy()]
    text_tokens = preprocess_text(texts, tokenizer)
    return jnp.array(images), jnp.array(text_tokens)


@nnx.jit
def create_sharded_model_and_optimizer():
    """Create and shard the CLIP model and optimizer following FSDP pattern."""
    model = CLIP.from_pretrained(HF_MODEL_NAME, use_pytorch=True, rngs=nnx.Rngs(0))
    state = nnx.state(model)
    pspecs = get_fsdp_sharding_specs(state, mesh, fsdp_axis_name="model")
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state)

    optimizer = nnx.Optimizer(model, optax.adam(LEARNING_RATE))
    state = nnx.state(optimizer)
    pspecs = get_fsdp_sharding_specs(state, mesh, fsdp_axis_name="model")
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(optimizer, sharded_state)

    return model, optimizer


def main() -> None:
    """Main training function."""
    global mesh
    devices = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(devices, ("model"))

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    train_dataset = create_synthetic_dataset(4096)
    train_dataset = train_dataset.batch(GLOBAL_BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    with mesh:
        model, optimizer = create_sharded_model_and_optimizer()

        model_spec = get_fsdp_sharding_specs(nnx.state(model), mesh, fsdp_axis_name="model")
        optimizer_spec = get_fsdp_sharding_specs(nnx.state(optimizer), mesh, fsdp_axis_name="model")

        in_shardings = (
            nnx.StateSharding(model_spec),
            nnx.StateSharding(optimizer_spec),
            P("model", None, None, None),  # images
            P("model", None),  # texts
        )
        out_shardings = (P(), {"accuracy": P(), "logit_scale": P()})

        train_step = nnx.jit(nnx.shard_map(train_fn, mesh=mesh, in_specs=in_shardings, out_specs=out_shardings))

        for epoch in range(NUM_EPOCHS):
            model.train()
            losses = []

            for step, batch in enumerate(train_dataset.take(100)):
                images, texts = preprocess_batch(batch, tokenizer)
                loss, metrics = train_step(model, optimizer, images, texts)
                losses.append(float(loss))
                print(loss)

            avg_loss = sum(losses) / len(losses)
            print(f"Epoch {epoch + 1} completed. Avg Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()
