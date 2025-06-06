import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike
from safetensors.flax import load_file


class TransformerEncoder(nnx.Module):
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
        self.norm2 = nnx.LayerNorm(hidden_size, dtype=dtype, param_dtype=param_dtype, rngs=rngs)

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
            dtype=dtype,
            param_dtype=param_dtype,
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
        cls_token = jnp.tile(self.cls_token.value, [batch_size, 1, 1])
        x = jnp.concat([cls_token, patches], axis=1)
        embeddings = x + self.position_embeddings.value
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
        mapping_list.extend(
            [
                (("patch_embeddings", "kernel"), ("vit", "embeddings", "patch_embeddings", "projection", "weight")),
                (("patch_embeddings", "bias"), ("vit", "embeddings", "patch_embeddings", "projection", "bias")),
            ]
        )
        mapping_list.extend([(("classifier", "kernel"), ("classifier", "weight")), (("classifier", "bias"), ("classifier", "bias"))])
        mapping_list.extend([(("final_norm", "scale"), ("vit", "layernorm", "weight")), (("final_norm", "bias"), ("vit", "layernorm", "bias"))])

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
