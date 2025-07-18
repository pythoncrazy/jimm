# SigLIP (Sigmoid-based Language Image Pre-training)

SigLIP (Sigmoid-based Language Image Pre-training) is a vision-language model that builds upon the principles of CLIP but introduces a key architectural change: it uses a sigmoid loss function instead of the softmax-based contrastive loss. Additionally, there are some slight implementation differences (no attention_mask for the text encoder, padding the text inputs, multihead attention pooling for the vision encoder rather than a linear projection layer).

This modification simplifies the training objective by treating the problem as a binary classification for each image-text pair (i.e., are they a positive or negative match?). This approach avoids the need for a global normalization over all pairs in a batch, which makes it more scalable and robust to noisy, web-scale data.

Key features of SigLIP:
1.  **Vision Encoder**: A Vision Transformer (ViT) with a Multi-Head Attention Pooling (MAP) head.
2.  **Text Encoder**: A standard Transformer model.
3.  **Sigmoid Loss**: Enables training on larger batches and noisier datasets without requiring careful data curation or complex negative sampling strategies.

SigLIP was introduced in the paper ["Sigmoid Loss for Language Image Pre-Training"](https://arxiv.org/abs/2303.15343) and has demonstrated improved performance and training efficiency.

::: jimm.models.siglip.SigLIP
    options:
        show_root_heading: true
        show_source: true
