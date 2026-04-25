from __future__ import annotations

from typing import Protocol

import torch


class EmbeddingModelProtocol(Protocol):
    def encode_summary(self, vectors: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return one summary embedding per batch item."""


class RelevancyModelProtocol(Protocol):
    def score_heap(self, summary: torch.Tensor, heap_vectors: torch.Tensor) -> torch.Tensor:
        """Return one scalar score per heap vector [batch, heap_size]."""
