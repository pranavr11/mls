from __future__ import annotations

import torch

from .embedding_autoencoder import EmbeddingAutoencoder
from .interfaces import EmbeddingModelProtocol
from .lightweight_embedding import MeanBottleneckEmbedding, MeanIdentityEmbedding


def build_embedding_model(
    *,
    model_type: str,
    hidden_size: int,
    latent_size: int,
    max_slots: int,
) -> torch.nn.Module | EmbeddingModelProtocol:
    t = str(model_type).lower()
    if t == "autoencoder":
        return EmbeddingAutoencoder(
            hidden_size=hidden_size,
            latent_size=latent_size,
            max_slots=max_slots,
        )
    if t == "mean_identity":
        return MeanIdentityEmbedding(hidden_size=hidden_size)
    if t == "mean_bottleneck":
        return MeanBottleneckEmbedding(hidden_size=hidden_size, latent_size=latent_size)
    raise ValueError(f"unsupported embedding model_type={model_type!r}")

