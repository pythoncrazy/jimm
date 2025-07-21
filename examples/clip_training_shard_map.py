from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
from flax import nnx
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int
from transformers import AutoTokenizer

from jimm.models.clip import CLIP

tf.config.set_visible_devices([], "GPU")


GLOBAL_BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4
MAX_SEQ_LENGTH = 77
IMAGE_SIZE = 224

HF_MODEL_NAME = "openai/clip-vit-base-patch32"

mesh = None


def named_sharding(*names: str | None) -> NamedSharding:
    """Helper function to create NamedSharding with the global mesh."""
    return NamedSharding(mesh, P(*names))


def preprocess_text(texts: list[str], tokenizer, max_length: int = MAX_SEQ_LENGTH):
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


def preprocess_images(images):
    """Preprocess images to [-1, 1] range.

    Args:
        images: Raw image array

    Returns:
        numpy array: Normalized images
    """
    if images.shape[-1] != 3:
        images = np.transpose(images, (0, 2, 3, 1))
    return (images.astype("float32") / 255.0) * 2.0 - 1.0


def clip_loss_fn(image_features: Float[Array, "local_batch embed_dim"], text_features: Float[Array, "local_batch embed_dim"], logit_scale: Float[Array, ""]) -> Float[Array, ""]:
    """Compute CLIP contrastive loss with all_gather for global embeddings.

    Args:
        image_features: Local image features per device
        text_features: Local text features per device
        logit_scale: Learnable temperature parameter

    Returns:
        Float[Array, ""]: Contrastive loss
    """
    image_features = image_features / jnp.linalg.norm(image_features, axis=-1, keepdims=True)
    text_features = text_features / jnp.linalg.norm(text_features, axis=-1, keepdims=True)

    all_image_features = jax.lax.all_gather(image_features, axis_name="model", axis=0)
    all_text_features = jax.lax.all_gather(text_features, axis_name="model", axis=0)

    global_batch_size = all_image_features.shape[0] * all_image_features.shape[1]
    embed_dim = all_image_features.shape[2]
    all_image_features = all_image_features.reshape(global_batch_size, embed_dim)
    all_text_features = all_text_features.reshape(global_batch_size, embed_dim)

    logits = jnp.exp(logit_scale) * all_image_features @ all_text_features.T
    labels = jnp.arange(global_batch_size)

    image_loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    text_loss = optax.softmax_cross_entropy_with_integer_labels(logits.T, labels)

    loss = (image_loss.mean() + text_loss.mean()) / 2.0
    return loss


def compute_loss_and_metrics(model: CLIP, images: Float[Array, "local_batch height width channels"], texts: Int[Array, "local_batch seq_len"]) -> Tuple[Float[Array, ""], Dict[str, Float[Array, ""]]]:
    """Compute loss and accuracy metrics.

    Args:
        model: CLIP model
        images: Local batch of images
        texts: Local batch of text tokens

    Returns:
        Tuple of loss and metrics dictionary. Loss is computed on the global batch, accuracy is computed on the local batch.
    """
    image_features = model.encode_image(images)
    text_features = model.encode_text(texts)

    loss = clip_loss_fn(image_features, text_features, model.logit_scale.value)

    # For accuracy, compute on local features (each device computes local accuracy)
    image_features_norm = image_features / jnp.linalg.norm(image_features, axis=-1, keepdims=True)
    text_features_norm = text_features / jnp.linalg.norm(text_features, axis=-1, keepdims=True)
    local_logits = jnp.exp(model.logit_scale.value) * image_features_norm @ text_features_norm.T
    local_predictions = jnp.argmax(local_logits, axis=-1)
    local_labels = jnp.arange(images.shape[0])
    local_accuracy = jnp.mean(local_predictions == local_labels)

    return loss, {"accuracy": local_accuracy, "logit_scale": model.logit_scale.value}


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


def preprocess_batch(batch: Dict[str, tf.Tensor], tokenizer):
    """Preprocess batch data to numpy arrays.

    Args:
        batch: TensorFlow batch dictionary
        tokenizer: Text tokenizer

    Returns:
        Tuple of numpy arrays (images, text_tokens)
    """
    images = preprocess_images(batch["image"].numpy())
    texts = [text.decode("utf-8") for text in batch["text"].numpy()]
    text_tokens = preprocess_text(texts, tokenizer)

    return images, text_tokens


def create_model_and_optimizer():
    """Create the CLIP model and optimizer."""
    model = CLIP.from_pretrained(HF_MODEL_NAME, use_pytorch=True)
    optimizer = nnx.Optimizer(model, optax.adam(LEARNING_RATE))
    return model, optimizer


def train_step_fn(model: CLIP, optimizer: nnx.Optimizer, images_np, texts_np):
    """Training step using shard_map with explicit communication."""
    images = jnp.array(images_np)
    texts = jnp.array(texts_np)

    grad_fn = nnx.value_and_grad(compute_loss_and_metrics, has_aux=True)
    (loss, metrics), grads = grad_fn(model, images, texts)

    grads = jax.tree.map(lambda x: jax.lax.psum(x, "model"), grads)
    optimizer.update(grads)

    loss = jax.lax.psum(loss, "model")
    metrics = jax.tree.map(lambda x: jax.lax.psum(x, "model"), metrics)

    return loss, metrics


def main() -> None:
    """Main training function."""
    global mesh
    devices = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(devices, ("model",))

    model, optimizer = create_model_and_optimizer()

    model_spec = nnx.get_partition_spec(nnx.state(model))
    optimizer_spec = nnx.get_partition_spec(nnx.state(optimizer))

    train_step = nnx.jit(
        nnx.shard_map(
            train_step_fn,
            mesh=mesh,
            in_specs=(
                nnx.StateSharding(model_spec),
                nnx.StateSharding(optimizer_spec),
                P("model", None, None, None),
                P("model", None),
            ),
            out_specs=(P(), {"accuracy": P(), "logit_scale": P()}),
        )
    )

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)

    train_dataset = create_synthetic_dataset(5000)
    train_dataset = train_dataset.batch(GLOBAL_BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        losses = []

        for step, batch in enumerate(train_dataset.take(100)):
            images_np, texts_np = preprocess_batch(batch, tokenizer)
            loss, metrics = train_step(model, optimizer, images_np, texts_np)
            losses.append(float(loss))

            if step % 20 == 0:
                print(f"Epoch {epoch + 1}, Step {step}: Loss={loss:.4f}, Acc={metrics['accuracy']:.4f}")

        avg_loss = sum(losses) / len(losses)
        print(f"Epoch {epoch + 1} completed. Avg Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()
