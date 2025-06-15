# This vit training script achieves a performance of 97.42% on the MNIST dataset, could probably be improved by using more layers and heads, but this is good enough to show that this works!

import jax
import jax.numpy as jnp
import optax
import tensorflow_datasets as tfds
import tensorflow as tf
from flax import nnx
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int
from typing import Dict, List, Tuple

from jimm.models.vit import VisionTransformer

IMG_SIZE_CONST: int = 28
PATCH_SIZE_CONST: int = 7
NUM_CLASSES_CONST: int = 10
IN_CHANNELS_CONST: int = 1
GLOBAL_BATCH_SIZE_CONST: int = 64
NUM_EPOCHS_CONST: int = 5
LEARNING_RATE_CONST: float = 1e-4

VIT_HIDDEN_SIZE_CONST: int = 512
VIT_NUM_LAYERS_CONST: int = 2
VIT_NUM_HEADS_CONST: int = 32
VIT_MLP_DIM_CONST: int = VIT_HIDDEN_SIZE_CONST * 4


def preprocess_batch(
    batch_tf: Dict[str, Array],
    mesh: Mesh,
) -> Tuple[Float[Array, "batch H W C"], Int[Array, " batch "]]:
    """Converts and shards a batch from TensorFlow Datasets.

    Args:
        batch_tf: A dictionary containing 'image' and 'label' np.ndarray.
        mesh: The JAX device mesh for sharding.

    Returns:
        A tuple of sharded JAX arrays (images, labels).
    """
    images_np = jnp.array(batch_tf["image"], dtype=jnp.float32) / 255.0
    labels_np = jnp.array(batch_tf["label"], dtype=jnp.int32)

    if images_np.ndim == 3:
        images_np = images_np[..., None]

    assert images_np.shape[0] == GLOBAL_BATCH_SIZE_CONST
    assert images_np.shape[1:] == (IMG_SIZE_CONST, IMG_SIZE_CONST, IN_CHANNELS_CONST)
    assert labels_np.shape[0] == GLOBAL_BATCH_SIZE_CONST

    sharded_images = jax.device_put(images_np, NamedSharding(mesh, P("data", None, None, None)))
    sharded_labels = jax.device_put(labels_np, NamedSharding(mesh, P("data")))
    return sharded_images, sharded_labels


def compute_loss_and_accuracy(
    model: VisionTransformer,
    images: Float[Array, "batch H W C"],
    labels: Int[Array, " batch "],
) -> Tuple[Float[Array, ""], Float[Array, ""]]:
    """Computes loss and accuracy for the given model and batch.

    Args:
        model: The VisionTransformer model.
        images: A batch of input images.
        labels: Corresponding labels for the images.

    Returns:
        A tuple containing the mean loss and accuracy.
    """
    logits = model(images)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return loss, accuracy


@nnx.jit
def train_step(
    model: VisionTransformer,
    optimizer: nnx.Optimizer,
    images: Float[Array, "batch H W C"],
    labels: Int[Array, " batch "],
) -> Tuple[Float[Array, ""], Float[Array, ""]]:
    """Performs a single training step, including gradient computation and optimizer update.

    Args:
        model: The VisionTransformer model.
        optimizer: The Optax optimizer wrapped in nnx.Optimizer.
        images: A batch of input images.
        labels: Corresponding labels for the images.

    Returns:
        A tuple containing the mean loss and accuracy for the batch.
    """
    grad_fn = nnx.value_and_grad(compute_loss_and_accuracy, has_aux=True)
    (loss, accuracy), grads = grad_fn(model, images, labels)
    optimizer.update(grads)
    return loss, accuracy


@nnx.jit
def eval_step(
    model: VisionTransformer,
    images: Float[Array, "batch H W C"],
    labels: Int[Array, " batch "],
) -> Tuple[Float[Array, ""], Float[Array, ""]]:
    """Performs a single evaluation step.

    Args:
        model: The VisionTransformer model.
        images: A batch of input images.
        labels: Corresponding labels for the images.

    Returns:
        A tuple containing the mean loss and accuracy for the batch.
    """
    return compute_loss_and_accuracy(model, images, labels)


@nnx.jit
def evaluate(
    model: VisionTransformer,
    images: Float[Array, "batch H W C"],
    labels: Int[Array, " batch "],
) -> Tuple[Float[Array, ""], Float[Array, ""]]:
    """Evaluate the model on a batch of images and labels.

    Args:
        model: The VisionTransformer model.
        images: A batch of input images.
        labels: Corresponding labels for the images.

    Returns:
        A tuple containing the loss and accuracy for this batch.
    """
    model.eval()
    return eval_step(model, images, labels)


def evaluate_dataset(
    model: VisionTransformer,
    dataset: tf.data.Dataset,
    mesh: Mesh,
) -> Tuple[float, float]:
    """Evaluates the model on an entire dataset.

    Args:
        model: The VisionTransformer model.
        dataset: The dataset to evaluate on.
        mesh: The JAX device mesh for sharding.

    Returns:
        A tuple containing the mean loss and accuracy across all batches.
    """
    model.eval()

    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0

    for batch_tf in tfds.as_numpy(dataset):
        images, labels = preprocess_batch(batch_tf, mesh)
        loss, accuracy = evaluate(model, images, labels)
        total_loss += loss.item()
        total_accuracy += accuracy.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0.0

    return avg_loss, avg_accuracy


def main() -> None:
    """Main function to run the MNIST ViT training."""
    num_devices = jax.local_device_count()
    device_mesh_shape = (num_devices, 1)
    device_mesh = mesh_utils.create_device_mesh(device_mesh_shape)
    mesh = Mesh(devices=device_mesh, axis_names=("data", "model"))

    rng_key_params = jax.random.PRNGKey(0)
    rng_key_dropout = jax.random.PRNGKey(1)

    model = VisionTransformer(
        num_classes=NUM_CLASSES_CONST,
        in_channels=IN_CHANNELS_CONST,
        img_size=IMG_SIZE_CONST,
        patch_size=PATCH_SIZE_CONST,
        num_layers=VIT_NUM_LAYERS_CONST,
        num_heads=VIT_NUM_HEADS_CONST,
        mlp_dim=VIT_MLP_DIM_CONST,
        hidden_size=VIT_HIDDEN_SIZE_CONST,
        dropout_rate=0.1,
        rngs=nnx.Rngs(params=rng_key_params, dropout=rng_key_dropout),
        mesh=mesh,
    )

    optimizer_def = optax.adam(learning_rate=LEARNING_RATE_CONST)
    optimizer = nnx.Optimizer(model, optimizer_def)

    train_ds = tfds.load("mnist", split="train[:80%]", as_supervised=False, shuffle_files=True)
    train_ds = train_ds.shuffle(10_000).batch(GLOBAL_BATCH_SIZE_CONST, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    val_ds = tfds.load("mnist", split="train[80%:]", as_supervised=False)
    val_ds = val_ds.batch(GLOBAL_BATCH_SIZE_CONST, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    test_ds = tfds.load("mnist", split="test", as_supervised=False)
    test_ds = test_ds.batch(GLOBAL_BATCH_SIZE_CONST, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    for epoch in range(NUM_EPOCHS_CONST):
        model.train()
        train_losses: List[float] = []
        train_accuracies: List[float] = []

        for step, batch_tf in enumerate(tfds.as_numpy(train_ds)):
            images, labels = preprocess_batch(batch_tf, mesh)
            loss, accuracy = train_step(model, optimizer, images, labels)
            train_losses.append(loss.item())
            train_accuracies.append(accuracy.item())

            if step % 50 == 0:
                print(f"Epoch {epoch + 1}/{NUM_EPOCHS_CONST}, Step {step}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")

        val_loss, val_accuracy = evaluate_dataset(model, val_ds, mesh)

        avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else 0
        avg_train_accuracy = sum(train_accuracies) / len(train_accuracies) if train_accuracies else 0
        print(f"End of Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}, Training Accuracy: {avg_train_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    test_loss, test_accuracy = evaluate_dataset(model, test_ds, mesh)
    print(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
