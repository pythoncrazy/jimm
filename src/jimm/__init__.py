from importlib.metadata import version

__version__ = version("jax-image-models")

from .common.autotuning import AutotuningResult, autotune, autotuned_fn, cached_autotune, load_autotune_result
from .common.tokamax_attention import make_tokamax_attention
from .common.tokamax_attention import tokamax_attention_fn as tokamax_attention
from .models import (
    CLIP,
    CLIPTextModel,
    CLIPVisionModel,
    SigLIP,
    SigLIPTextModel,
    SigLIPVisionModel,
    VisionTransformer,
)

__all__ = [
    "tokamax_attention",
    "make_tokamax_attention",
    "autotune",
    "cached_autotune",
    "autotuned_fn",
    "load_autotune_result",
    "AutotuningResult",
    "VisionTransformer",
    "CLIP",
    "CLIPTextModel",
    "CLIPVisionModel",
    "SigLIP",
    "SigLIPTextModel",
    "SigLIPVisionModel",
]
