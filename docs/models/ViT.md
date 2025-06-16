# ViT (Vision Transformer)

The ViT (Vision Transformer) is a transformer-based neural network architecture for image classification. It divides an image into fixed-size patches, linearly embeds each patch, adds position embeddings, and processes the resulting sequence of vectors through a standard transformer encoder.

The ViT model was introduced in the paper ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929) and has shown strong performance on image classification benchmarks.

::: jimm.models.vit.VisionTransformer
    options:
        show_root_heading: true
        show_source: true