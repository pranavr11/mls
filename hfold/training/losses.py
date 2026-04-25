from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_reconstruction_loss(reconstructed: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    cosine = F.cosine_similarity(reconstructed, target, dim=-1)
    loss = 1.0 - cosine
    if mask is not None:
        weights = mask.to(loss.dtype)
        return (loss * weights).sum() / weights.sum().clamp_min(1.0)
    return loss.mean()


def ranking_loss(pred_scores: torch.Tensor, target_scores: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    pred_diff = pred_scores.unsqueeze(-1) - pred_scores.unsqueeze(-2)
    target_diff = target_scores.unsqueeze(-1) - target_scores.unsqueeze(-2)
    signs = torch.sign(target_diff)
    raw = -signs * pred_diff + margin
    return torch.relu(raw).mean()
