from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class HiddenStateSample:
    backbone: str
    evicted_vectors: torch.Tensor
    heap_vectors: torch.Tensor
    teacher_scores: torch.Tensor


class SyntheticHiddenStateDataset(Dataset):
    """
    Placeholder dataset producing synthetic hidden-state tuples.
    Can be replaced by real extractor outputs without changing trainers.
    """

    def __init__(
        self,
        *,
        size: int,
        backbone: str,
        hidden_size: int,
        max_heap_size: int,
        seed: int = 0,
    ) -> None:
        self.size = size
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.max_heap_size = max_heap_size
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> HiddenStateSample:
        del index
        evicted = torch.randn(self.max_heap_size, self.hidden_size, generator=self.generator)
        heap = torch.randn(self.max_heap_size, self.hidden_size, generator=self.generator)
        teacher_scores = torch.softmax(torch.randn(self.max_heap_size, generator=self.generator), dim=0)
        return HiddenStateSample(
            backbone=self.backbone,
            evicted_vectors=evicted,
            heap_vectors=heap,
            teacher_scores=teacher_scores,
        )
