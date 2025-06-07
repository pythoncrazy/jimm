# Jax Image Modeling of Models (jimm)
This aims to be the jax counterpart to timm, with the exception that for image-text models (CLIP, SigLIP, etc), we support the text model entirely.
Made with flax nnx, supports weight loading from pytorch_model.bin and safetensors (as well as both methods from huggingface).