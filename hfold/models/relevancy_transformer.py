from __future__ import annotations

import torch
from torch import nn

from .interfaces import RelevancyModelProtocol


class RelevancyTransformer(nn.Module, RelevancyModelProtocol):
    """
    Encoder-only transformer over [global_summary, heap_vectors...].
    """

    def __init__(self, hidden_size: int, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.score_head = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.GELU(), nn.Linear(hidden_size, 1))

    def score_heap(self, summary: torch.Tensor, heap_vectors: torch.Tensor) -> torch.Tensor:
        summary_token = summary.unsqueeze(1)
        sequence = torch.cat([summary_token, heap_vectors], dim=1)
        encoded = self.encoder(sequence)
        summary_encoded = encoded[:, :1, :].expand(-1, heap_vectors.size(1), -1)
        heap_encoded = encoded[:, 1:, :]
        features = torch.cat([summary_encoded, heap_encoded], dim=-1)
        scores = self.score_head(features).squeeze(-1)
        return scores

    def forward(self, summary: torch.Tensor, heap_vectors: torch.Tensor) -> torch.Tensor:
        return self.score_heap(summary, heap_vectors)
