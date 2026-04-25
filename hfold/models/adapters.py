from __future__ import annotations

import torch
from torch import nn


class BackboneAdapter(nn.Module):
    def __init__(self, in_dim: int, shared_dim: int) -> None:
        super().__init__()
        self.to_shared = nn.Linear(in_dim, shared_dim)
        self.from_shared = nn.Linear(shared_dim, in_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_shared(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.from_shared(x)


class BackboneAdapterRegistry(nn.Module):
    def __init__(self, specs: dict[str, int], shared_dim: int) -> None:
        super().__init__()
        if not specs:
            raise ValueError("specs cannot be empty")
        self.adapters = nn.ModuleDict(
            {name: BackboneAdapter(in_dim=in_dim, shared_dim=shared_dim) for name, in_dim in specs.items()}
        )

    def encode(self, backbone: str, x: torch.Tensor) -> torch.Tensor:
        return self.adapters[backbone].encode(x)

    def decode(self, backbone: str, x: torch.Tensor) -> torch.Tensor:
        return self.adapters[backbone].decode(x)
