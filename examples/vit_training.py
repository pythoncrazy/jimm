import jax
import jax.numpy as jnp
import optax
import tensorflow_datasets as tfds
from flax import nnx
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int
from typing import Dict, Tuple

from jimm.models.vit import VisionTransformer

# Configuration
IMG_SIZE_CONST: int = 28
PATCH_SIZE_CONST: int = 7
NUM_CLASSES_CONST: int = 10
IN_CHANNELS_CONST: int = 1
GLOBAL_BATCH_SIZE_CONST: int = 64
NUM_EPOCHS_CONST: int = 5
LEARNING_RATE_CONST: float = 1e-3

VIT_HIDDEN_SIZE_CONST: int = 192
VIT_NUM_LAYERS_CONST: int = 4
VIT_NUM_HEADS_CONST: int = 3
VIT_MLP_DIM_CONST: int = VIT_HIDDEN_SIZE_CONST * 4


def preprocess_batch(
    batch_tf: Dict[str, Array],
    mesh: Mesh,
) -> Tuple[Float[Array, "batch H W C"], Int[Array, "batch"]]:
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
    labels: Int[Array, "batch"],
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
    labels: Int[Array, "batch"],
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
        mesh=mesh
    )
    model.train()

    optimizer_def = optax.adam(learning_rate=LEARNING_RATE_CONST)
    optimizer = nnx.Optimizer(model, optimizer_def)

    train_ds = tfds.load('mnist', split='train', as_supervised=False, shuffle_files=True)
    train_ds = train_ds.shuffle(10_000).batch(GLOBAL_BATCH_SIZE_CONST).prefetch(tfds.AUTOTUNE)

    for epoch in range(NUM_EPOCHS_CONST):
        for step, batch_tf in enumerate(tfds.as_numpy(train_ds)):
            images, labels = preprocess_batch(batch_tf, mesh)
            loss, accuracy = train_step(model, optimizer, images, labels)
            if step % 50 == 0:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS_CONST}, Step {step}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")
        print(f"End of Epoch {epoch+1}, Final Batch Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")

if __name__ == "__main__":
    main()
