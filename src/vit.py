# Largely Inspired by https://docs.jaxstack.ai/en/latest/JAX_Vision_transformer.html

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import requests
from flax import nnx
from jax.typing import DTypeLike
from PIL import Image
from safetensors.flax import load_file
from transformers import ViTForImageClassification, ViTImageProcessor


class TransformerEncoder(nnx.Module):
    """
    A single transformer encoder block in the ViT model, inheriting from `flax.nnx.Module`.

    Args:
        hidden_size (int): Input/output embedding dimensionality.
        mlp_dim (int): Dimension of the feed-forward/MLP block hidden layer.
        num_heads (int): Number of attention heads.
        dropout_rate (float): Dropout rate. Defaults to 0.0.
        rngs (flax.nnx.Rngs): A set of named `flax.nnx.RngStream` objects that generate a stream of JAX pseudo-random number generator (PRNG) keys. Defaults to `flax.nnx.Rngs(0)`.
    """

    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        dtype: DTypeLike = jnp.bfloat16,
        param_dtype: DTypeLike = jnp.bfloat16,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ) -> None:
        self.norm1 = nnx.LayerNorm(hidden_size, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_size,
            dropout_rate=dropout_rate,
            broadcast_dropout=False,
            decode=False,
            deterministic=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(hidden_size, rngs=rngs)

        self.mlp = nnx.Sequential(
            nnx.Linear(hidden_size, mlp_dim, dtype=dtype, param_dtype=param_dtype, rngs=rngs),
            nnx.gelu,
            nnx.Dropout(dropout_rate, rngs=rngs),
            nnx.Linear(mlp_dim, hidden_size, dtype=dtype, param_dtype=param_dtype, rngs=rngs),
            nnx.Dropout(dropout_rate, rngs=rngs),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nnx.Module):
    """Implements the ViT model, inheriting from `flax.nnx.Module`.

    Args:
        num_classes (int): Number of classes in the classification. Defaults to 1000.
        in_channels (int): Number of input channels in the image (such as 3 for RGB). Defaults to 3.
        img_size (int): Input image size. Defaults to 224.
        patch_size (int): Size of the patches extracted from the image. Defaults to 16.
        num_layers (int): Number of transformer encoder layers. Defaults to 12.
        num_heads (int): Number of attention heads in each transformer layer. Defaults to 12.
        mlp_dim (int): Dimension of the hidden layers in the feed-forward/MLP block. Defaults to 3072.
        hidden_size (int): Dimensionality of the embedding vectors. Defaults to 3072.
        dropout_rate (int): Dropout rate (for regularization). Defaults to 0.1.
        rngs (flax.nnx.Rngs): A set of named `flax.nnx.RngStream` objects that generate a stream of JAX pseudo-random number generator (PRNG) keys. Defaults to `flax.nnx.Rngs(0)`.

    """

    def __init__(
        self,
        num_classes: int = 1000,
        in_channels: int = 3,
        img_size: int = 224,
        patch_size: int = 16,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        hidden_size: int = 768,
        dropout_rate: float = 0.1,
        dtype: DTypeLike = jnp.bfloat16,
        param_dtype: DTypeLike = jnp.bfloat16,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        n_patches = (img_size // patch_size) ** 2
        self.patch_embeddings = nnx.Conv(
            in_channels,
            hidden_size,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            use_bias=True,
            rngs=rngs,
        )
        initializer = jax.nn.initializers.truncated_normal(stddev=0.02)
        self.position_embeddings = nnx.Param(initializer(rngs.params(), (1, n_patches + 1, hidden_size), dtype=dtype))
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
        self.cls_token = nnx.Param(jnp.zeros((1, 1, hidden_size), dtype=dtype))
        self.encoder = nnx.Sequential(
            *[
                TransformerEncoder(
                    hidden_size,
                    mlp_dim,
                    num_heads,
                    dropout_rate,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    rngs=rngs,
                )
                for i in range(num_layers)
            ]
        )
        self.final_norm = nnx.LayerNorm(hidden_size, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.classifier = nnx.Linear(hidden_size, num_classes, dtype=dtype, param_dtype=param_dtype, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        patches = self.patch_embeddings(x)
        batch_size = patches.shape[0]
        patches = patches.reshape(batch_size, -1, patches.shape[-1])
        cls_token = jnp.tile(self.cls_token, [batch_size, 1, 1])
        x = jnp.concat([cls_token, patches], axis=1)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        x = self.encoder(embeddings)
        x = self.final_norm(x)
        x = x[:, 0]
        return self.classifier(x)


x = jnp.ones((4, 224, 224, 3))
pytorch_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
model = VisionTransformer(
    num_classes=1000,
)
y = model(x)
print("Predictions shape: ", y.shape)
loaded = load_file("weights/model.safetensors")
print("Loaded safetensors keys (overview):", loaded.keys())


def vit_inplace_copy_weights(*, params, dst_model):
    assert isinstance(dst_model, VisionTransformer)

    params_fstate = params

    print("\nSafetensors parameters (loaded from file):")
    for path_str, tensor_val in params_fstate.items():
        print(f'  Path: "{path_str}", Shape: {tensor_val.shape}')
    print("-" * 30)

    print("Flax NNX Model parameters (initial):")
    flax_model_params = nnx.state(dst_model, nnx.Param)
    flax_model_params_fstate = dict(flax_model_params.flat_state())
    for path, param_obj in flax_model_params_fstate.items():
        print(f"  Path: {path}, Shape: {param_obj.value.shape}")
    print("-" * 30)

    def hf_param_name(name: str) -> str:
        return "weight" if name in ["kernel", "scale"] else name

    num_encoder_layers = 12
    num_heads = 12
    hidden_size_per_head = 64
    hidden_size = num_heads * hidden_size_per_head
    # can you refactor the following mapping to be more concise, while still maintaining all of the information? Make sure that the code that you generate is short and concise, with absolutely no comments whatsoever ai!
    params_name_mapping = {
        ("cls_token",): ("vit", "embeddings", "cls_token"),
        ("position_embeddings",): ("vit", "embeddings", "position_embeddings"),
        **{
            ("patch_embeddings", flax_p_name): (
                "vit",
                "embeddings",
                "patch_embeddings",
                "projection",
                hf_param_name(flax_p_name),
            )
            for flax_p_name in ["kernel", "bias"]
        },
        **{
            ("encoder", "layers", i, "attn", y_type, flax_p_name): (
                "vit",
                "encoder",
                "layer",
                str(i),
                "attention",
                "attention",
                y_type,
                hf_param_name(flax_p_name),
            )
            for flax_p_name in ["kernel", "bias"]
            for y_type in ["key", "value", "query"]
            for i in range(num_encoder_layers)
        },
        **{
            ("encoder", "layers", i, "attn", "out", flax_p_name): (
                "vit",
                "encoder",
                "layer",
                str(i),
                "attention",
                "output",
                "dense",
                hf_param_name(flax_p_name),
            )
            for flax_p_name in ["kernel", "bias"]
            for i in range(num_encoder_layers)
        },
        **{
            ("encoder", "layers", i, "mlp", "layers", y1_idx, flax_p_name): (
                "vit",
                "encoder",
                "layer",
                str(i),
                y2_name,
                "dense",
                hf_param_name(flax_p_name),
            )
            for flax_p_name in ["kernel", "bias"]
            for y1_idx, y2_name in [(0, "intermediate"), (3, "output")]
            for i in range(num_encoder_layers)
        },
        **{
            ("encoder", "layers", i, y1_norm_name, flax_p_name): (
                "vit",
                "encoder",
                "layer",
                str(i),
                y2_hf_norm_name,
                hf_param_name(flax_p_name),
            )
            for flax_p_name in ["scale", "bias"]
            for y1_norm_name, y2_hf_norm_name in [
                ("norm1", "layernorm_before"),
                ("norm2", "layernorm_after"),
            ]
            for i in range(num_encoder_layers)
        },
        **{
            ("final_norm", flax_p_name): (
                "vit",
                "layernorm",
                hf_param_name(flax_p_name),
            )
            for flax_p_name in ["scale", "bias"]
        },
        **{("classifier", flax_p_name): ("classifier", hf_param_name(flax_p_name)) for flax_p_name in ["kernel", "bias"]},
    }

    nonvisited = set(flax_model_params_fstate.keys())

    for flax_dst_key_tuple, hf_src_key_tuple in params_name_mapping.items():
        assert flax_dst_key_tuple in flax_model_params_fstate, flax_dst_key_tuple

        hf_src_key_as_string = ".".join(hf_src_key_tuple)
        assert hf_src_key_as_string in params_fstate, f"HF key '{hf_src_key_as_string}' (from Flax key {flax_dst_key_tuple}) not found in loaded safetensors."

        nonvisited.remove(flax_dst_key_tuple)

        src_value = params_fstate[hf_src_key_as_string]

        if flax_dst_key_tuple == ("patch_embeddings", "kernel"):
            # HF conv kernel: (out_channels, in_channels, kernel_height, kernel_width)
            # Flax nnx.Conv kernel: (kernel_height, kernel_width, in_channels, out_channels)
            src_value = jnp.transpose(src_value, (2, 3, 1, 0))
        elif hf_src_key_tuple[-1] == "weight" and hf_src_key_tuple[-2] in (
            "key",
            "value",
            "query",
        ):  # QKV weights
            # HF QKV weight: (num_heads * head_dim, hidden_size)
            # Flax nnx.MultiHeadAttention QKV weight: (hidden_size, num_heads, head_dim)
            src_value = jnp.transpose(src_value, (1, 0))  # Shape: (hidden_size, num_heads * head_dim)
            src_value = src_value.reshape((hidden_size, num_heads, hidden_size_per_head))
        elif hf_src_key_tuple[-1] == "bias" and hf_src_key_tuple[-2] in (
            "key",
            "value",
            "query",
        ):  # QKV biases
            # HF QKV bias: (num_heads * head_dim)
            # Flax nnx.MultiHeadAttention QKV bias: (num_heads, head_dim)
            src_value = src_value.reshape((num_heads, hidden_size_per_head))
        elif hf_src_key_tuple[-4:] == (
            "attention",
            "output",
            "dense",
            "weight",
        ):  # Attention output projection weight
            # HF attention output weight: (hidden_size, num_heads * head_dim)
            # Flax nnx.MultiHeadAttention output weight: (num_heads, head_dim, hidden_size)
            src_value = jnp.transpose(src_value, (1, 0))  # Shape: (num_heads * head_dim, hidden_size)
            src_value = src_value.reshape((num_heads, hidden_size_per_head, hidden_size))
        elif hf_src_key_tuple[-1] == "weight" and src_value.ndim == 2:  # General 2D Linear layer weights (MLP, Classifier)
            # HF Linear weight: (out_features, in_features)
            # Flax nnx.Linear kernel: (in_features, out_features)
            src_value = jnp.transpose(src_value, (1, 0))

        dst_value_obj = flax_model_params_fstate[flax_dst_key_tuple]
        assert src_value.shape == dst_value_obj.value.shape, f"Shape mismatch for {flax_dst_key_tuple} (Flax) vs {hf_src_key_as_string} (HF): {dst_value_obj.value.shape} != {src_value.shape}"
        dst_value_obj.value = src_value.copy()
        assert dst_value_obj.value.mean() == src_value.mean(), (
            dst_value_obj.value.mean(),
            src_value.mean(),
        )

    assert len(nonvisited) == 0, f"Some Flax model parameters were not visited: {nonvisited}"
    nnx.update(dst_model, nnx.State.from_flat_path(flax_model_params_fstate))


vit_inplace_copy_weights(params=loaded, dst_model=model)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

inputs = processor(images=image, return_tensors="pt")
outputs = pytorch_model(**inputs)
logits = outputs.logits


model.eval()
x = jnp.transpose(inputs["pixel_values"].detach().cpu().numpy(), axes=(0, 2, 3, 1))
output = model(x)

ref_class_idx = logits.argmax(-1).item()
pred_class_idx = output.argmax(-1).item()
print(jnp.abs(logits[0, :].detach().cpu().numpy() - output[0, :]).max())

fig, axs = plt.subplots(1, 2, figsize=(12, 8))
axs[0].set_title(f"Reference model:\n{pytorch_model.config.id2label[ref_class_idx]}\nP={nnx.softmax(logits.detach().cpu().numpy(), axis=-1)[0, ref_class_idx]}")
axs[0].imshow(image)
axs[1].set_title(f"Our model:\n{pytorch_model.config.id2label[pred_class_idx]}\nP={nnx.softmax(output, axis=-1)[0, pred_class_idx]}")
axs[1].imshow(image)
plt.savefig("tmp/plot.png")
