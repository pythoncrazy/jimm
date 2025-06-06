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
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
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
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
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

    @classmethod
    def from_pretrained(cls, params_path: str) -> tuple["VisionTransformer", int]:
        params_fstate = load_file(params_path)

        hidden_size = params_fstate["vit.embeddings.cls_token"].shape[-1]
        num_classes = params_fstate["classifier.bias"].shape[0]

        max_layer_idx = -1
        for k in params_fstate:
            if k.startswith("vit.encoder.layer."):
                max_layer_idx = max(max_layer_idx, int(k.split(".")[3]))

        num_layers = max_layer_idx + 1

        mlp_dim = params_fstate["vit.encoder.layer.0.intermediate.dense.weight"].shape[0]

        assumed_head_dim = 64
        num_heads = hidden_size // assumed_head_dim

        patch_kernel_shape = params_fstate["vit.embeddings.patch_embeddings.projection.weight"].shape
        patch_size = patch_kernel_shape[2]

        num_patches = params_fstate["vit.embeddings.position_embeddings"].shape[1] - 1
        img_size_dim = int(jnp.sqrt(num_patches))
        if img_size_dim * img_size_dim != num_patches:
            raise ValueError(f"num_patches {num_patches} is not a perfect square.")
        img_size = img_size_dim * patch_size

        model = cls(
            num_classes=num_classes,
            img_size=img_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            hidden_size=hidden_size,
        )

        flax_model_params_fstate = dict(nnx.state(model, nnx.Param).flat_state())

        def hf_param_name(name: str) -> str:
            return "weight" if name in ["kernel", "scale"] else name

        hidden_size_per_head = hidden_size // num_heads

        mapping_list = [
            (("cls_token",), ("vit", "embeddings", "cls_token")),
            (("position_embeddings",), ("vit", "embeddings", "position_embeddings")),
        ]
        mapping_list.extend([(("patch_embeddings", p_name), ("vit", "embeddings", "patch_embeddings", "projection", hf_param_name(p_name))) for p_name in ["kernel", "bias"]])
        mapping_list.extend([(("classifier", p_name), ("classifier", hf_param_name(p_name))) for p_name in ["kernel", "bias"]])
        mapping_list.extend([(("final_norm", p_name), ("vit", "layernorm", hf_param_name(p_name))) for p_name in ["scale", "bias"]])

        for i in range(num_layers):
            flax_base = ("encoder", "layers", i)
            hf_base = ("vit", "encoder", "layer", str(i))
            mapping_list.extend(
                [(flax_base + ("attn", y_type, p_name), hf_base + ("attention", "attention", y_type, hf_param_name(p_name))) for p_name in ["kernel", "bias"] for y_type in ["key", "value", "query"]]
            )
            mapping_list.extend([(flax_base + ("attn", "out", p_name), hf_base + ("attention", "output", "dense", hf_param_name(p_name))) for p_name in ["kernel", "bias"]])
            mapping_list.extend(
                [
                    (flax_base + ("mlp", "layers", y1_idx, p_name), hf_base + (y2_name, "dense", hf_param_name(p_name)))
                    for p_name in ["kernel", "bias"]
                    for y1_idx, y2_name in [(0, "intermediate"), (3, "output")]
                ]
            )
            mapping_list.extend(
                [
                    (flax_base + (norm_flax, p_name), hf_base + (norm_hf, hf_param_name(p_name)))
                    for p_name in ["scale", "bias"]
                    for norm_flax, norm_hf in [("norm1", "layernorm_before"), ("norm2", "layernorm_after")]
                ]
            )
        params_name_mapping = dict(mapping_list)
        nonvisited = set(flax_model_params_fstate.keys())

        for flax_dst_key_tuple, hf_src_key_tuple in params_name_mapping.items():
            assert flax_dst_key_tuple in flax_model_params_fstate, flax_dst_key_tuple
            hf_src_key_as_string = ".".join(hf_src_key_tuple)
            assert hf_src_key_as_string in params_fstate, f"HF key '{hf_src_key_as_string}' (from Flax key {flax_dst_key_tuple}) not found in loaded safetensors."
            nonvisited.remove(flax_dst_key_tuple)
            src_value = params_fstate[hf_src_key_as_string]

            if flax_dst_key_tuple == ("patch_embeddings", "kernel"):
                src_value = jnp.transpose(src_value, (2, 3, 1, 0))
            elif hf_src_key_tuple[-1] == "weight" and hf_src_key_tuple[-2] in ("key", "value", "query"):
                src_value = jnp.transpose(src_value, (1, 0))
                src_value = src_value.reshape((hidden_size, num_heads, hidden_size_per_head))
            elif hf_src_key_tuple[-1] == "bias" and hf_src_key_tuple[-2] in ("key", "value", "query"):
                src_value = src_value.reshape((num_heads, hidden_size_per_head))
            elif hf_src_key_tuple[-4:] == ("attention", "output", "dense", "weight"):
                src_value = jnp.transpose(src_value, (1, 0))
                src_value = src_value.reshape((num_heads, hidden_size_per_head, hidden_size))
            elif hf_src_key_tuple[-1] == "weight" and src_value.ndim == 2:
                src_value = jnp.transpose(src_value, (1, 0))

            dst_value_obj = flax_model_params_fstate[flax_dst_key_tuple]
            assert src_value.shape == dst_value_obj.value.shape, f"Shape mismatch for {flax_dst_key_tuple} (Flax) vs {hf_src_key_as_string} (HF): {dst_value_obj.value.shape} != {src_value.shape}"
            dst_value_obj.value = src_value.copy()
            assert dst_value_obj.value.mean() == src_value.mean(), (dst_value_obj.value.mean(), src_value.mean())

        assert len(nonvisited) == 0, f"Some Flax model parameters were not visited: {nonvisited}"
        nnx.update(model, nnx.State.from_flat_path(flax_model_params_fstate))
        return model, img_size


HF_MODEL_NAME = "google/vit-large-patch16-224"
SAFETENSORS_PATH = "weights/model.safetensors"

model, inferred_img_size = VisionTransformer.from_pretrained(SAFETENSORS_PATH)

x_dummy = jnp.ones((4, inferred_img_size, inferred_img_size, 3))
y_dummy = model(x_dummy)
print("Predictions shape: ", y_dummy.shape)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

processor = ViTImageProcessor.from_pretrained(HF_MODEL_NAME)
pytorch_model = ViTForImageClassification.from_pretrained(HF_MODEL_NAME)

inputs = processor(images=image, return_tensors="pt")
outputs = pytorch_model(**inputs)
logits_ref = outputs.logits

model.eval()
x_eval = jnp.transpose(inputs["pixel_values"].detach().cpu().numpy(), axes=(0, 2, 3, 1))
logits_flax = model(x_eval)

ref_class_idx = logits_ref.argmax(-1).item()
pred_class_idx = logits_flax.argmax(-1).item()
print(f"Max absolute difference: {jnp.abs(logits_ref[0, :].detach().cpu().numpy() - logits_flax[0, :]).max()}")

fig, axs = plt.subplots(1, 2, figsize=(12, 8))
axs[0].set_title(f"Reference model:\n{pytorch_model.config.id2label[ref_class_idx]}\nP={nnx.softmax(logits_ref.detach().cpu().numpy(), axis=-1)[0, ref_class_idx]}")
axs[0].imshow(image)
axs[1].set_title(f"Our model:\n{pytorch_model.config.id2label[pred_class_idx]}\nP={nnx.softmax(logits_flax, axis=-1)[0, pred_class_idx]}")
axs[1].imshow(image)
plt.savefig("tmp/plot.png")
