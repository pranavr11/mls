from __future__ import annotations

import torch
from torch import nn

from .interfaces import EmbeddingModelProtocol


def _masked_mean(vectors: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
    if padding_mask is None:
        return vectors.mean(dim=1)
    weights = padding_mask.to(vectors.dtype).unsqueeze(-1)
    return (vectors * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-6)


class MeanIdentityEmbedding(nn.Module, EmbeddingModelProtocol):
    """
    Cheapest summary model:
    - encode: masked mean over slots
    - decode: broadcast summary back to slots
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)

    def encode_summary(self, vectors: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        return _masked_mean(vectors, padding_mask=padding_mask)

    def decode_from_summary(self, summary: torch.Tensor, num_slots: int) -> torch.Tensor:
        return summary.unsqueeze(1).expand(-1, int(num_slots), -1)


class MeanBottleneckEmbedding(nn.Module, EmbeddingModelProtocol):
    """
    Lightweight bottleneck model:
    - encode: masked mean, then project to latent
    - decode: project latent back to hidden and broadcast to slots
    """

    def __init__(self, hidden_size: int, latent_size: int) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.latent_size = int(latent_size)
        self.down = nn.Linear(self.hidden_size, self.latent_size, bias=False)
        self.up = nn.Linear(self.latent_size, self.hidden_size, bias=False)
        self.norm = nn.LayerNorm(self.latent_size)

    def encode_summary(self, vectors: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        pooled = _masked_mean(vectors, padding_mask=padding_mask)
        return self.norm(self.down(pooled))

    def decode_from_summary(self, summary: torch.Tensor, num_slots: int) -> torch.Tensor:
        expanded = self.up(summary)
        return expanded.unsqueeze(1).expand(-1, int(num_slots), -1)

