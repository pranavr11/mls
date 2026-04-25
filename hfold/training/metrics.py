from __future__ import annotations

import torch


@torch.no_grad()
def cosine_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item())


@torch.no_grad()
def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.mse_loss(a, b).item())
