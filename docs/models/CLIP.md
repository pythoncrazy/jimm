# CLIP (Contrastive Language–Image Pre-training)

CLIP (Contrastive Language–Image Pre-training) is a neural network architecture that learns visual concepts from natural language supervision. It is trained on a large dataset of image-text pairs to create a unified vision-language model that can understand both images and text in a shared semantic space.

CLIP consists of two main components:
1. A vision encoder (Vision Transformer) that processes images into visual features
2. A text encoder (Transformer) that processes text into textual features

The model is trained using contrastive learning, where it learns to maximize the cosine similarity between the embeddings of matching image-text pairs while minimizing it for non-matching pairs. This allows CLIP to perform zero-shot classification by comparing image embeddings with text embeddings of potential labels.

CLIP was introduced in the paper ["Learning Transferable Visual Models From Natural Language Supervision"](https://arxiv.org/abs/2103.00020) and has shown remarkable zero-shot generalization capabilities across a wide range of visual classification tasks. The CLIP model combines a Vision Transformer and a Text Transformer to learn joint representations of images and text. It is trained to maximize the similarity between matching image-text pairs while minimizing similarity between non-matching pairs.

::: jimm.models.clip.CLIP
    options:
        show_root_heading: true
        show_source: true
