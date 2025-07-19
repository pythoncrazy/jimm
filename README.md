# Jax Image Modeling of Models (jimm)
Docs are at: [https://locamage.dev/jimm](https://locamage.dev/jimm)
- This aims to be the jax counterpart to timm, with the exception that for image-text models (CLIP, SigLIP, etc), we support the text model entirely.
- Made with flax nnx, supports weight loading from pytorch_model.bin and safetensors (as well as both methods from huggingface).

Models Supported:
- Vision Transformers
    - Both with a classification linear layer, or not
    - Using a CLS Token for pooling, or using Multihead Attention Pooling
    - Can load any standard variant of Vision Transformers of any size/resolution(e.g. "google/vit-base-patch16-224" or "google/vit-large-patch16-384")
- CLIP
    - Can load from any checkpoints of the clip model on github (such as "openai/clip-vit-base-patch32" or "geolocal/StreetCLIP")
- SigLIP
    - Can load any non-naflex version of the SigLIP model, from both siglipv1 and siglipv2 (eg "google/siglip-base-patch16-256" or "google/siglip2-large-patch16-512" from huggingface or locally)
## Installation
### Using pixi.sh:
`pixi add timm@https://github.com/Locamage/jimm.git --pypi`
### Using uv
`uv add --dev git+https://github.com/Locamage/jimm.git`
or if you prefer to not add as a direct dependency:
`uv pip install git+https://github.com/Locamage/jimm.git`
### Using pip/conda
`pip install git+https://github.com/Locamage.git`
