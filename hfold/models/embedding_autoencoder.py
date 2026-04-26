from __future__ import annotations

import torch
from torch import nn

from .interfaces import EmbeddingModelProtocol


class EmbeddingAutoencoder(nn.Module, EmbeddingModelProtocol):
    """
    Autoencoder over up to S vectors with a single-vector bottleneck.
    Input: [batch, slots, dim]
    """

    def __init__(self, hidden_size: int, latent_size: int, max_slots: int, num_layers: int = 2) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.max_slots = max_slots
        enc_layers: list[nn.Module] = [nn.Linear(hidden_size, hidden_size), nn.GELU()]
        for _ in range(max(0, num_layers - 1)):
            enc_layers.extend([nn.Linear(hidden_size, hidden_size), nn.GELU()])
        self.token_encoder = nn.Sequential(*enc_layers)
        self.summary_proj = nn.Linear(hidden_size, latent_size)
        self.summary_norm = nn.LayerNorm(latent_size)

        # Slot-aware decoder: a learnable query per slot is concatenated with
        # the bottleneck summary so reconstructions can differ across slots.
        self.slot_queries = nn.Parameter(torch.randn(max_slots, latent_size) * 0.02)
        self.slot_decoder = nn.Sequential(
            nn.Linear(latent_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def encode_summary(self, vectors: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.token_encoder(vectors)
        if padding_mask is not None:
            weights = padding_mask.to(encoded.dtype).unsqueeze(-1)
            pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-6)
        else:
            pooled = encoded.mean(dim=1)
        summary = self.summary_norm(self.summary_proj(pooled))
        return summary

    def decode_from_summary(self, summary: torch.Tensor, num_slots: int) -> torch.Tensor:
        if num_slots > self.max_slots:
            raise ValueError(
                f"requested {num_slots} reconstruction slots but max_slots={self.max_slots}"
            )
        slot_queries = self.slot_queries[:num_slots]
        batch_size = summary.size(0)
        summary_expanded = summary.unsqueeze(1).expand(-1, num_slots, -1)
        slot_queries_expanded = slot_queries.unsqueeze(0).expand(batch_size, -1, -1)
        decoder_input = torch.cat([summary_expanded, slot_queries_expanded], dim=-1)
        return self.slot_decoder(decoder_input)

    def forward(
        self,
        vectors: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        summary = self.encode_summary(vectors, padding_mask=padding_mask)
        reconstructed = self.decode_from_summary(summary, num_slots=vectors.size(1))
        return reconstructed, summary
