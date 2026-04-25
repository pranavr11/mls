from __future__ import annotations

import torch

from .hidden_state_dataset import HiddenStateSample


def collate_hfold_samples(samples: list[HiddenStateSample]) -> dict[str, torch.Tensor | list[str]]:
    backbones = [sample.backbone for sample in samples]
    evicted = torch.stack([sample.evicted_vectors for sample in samples], dim=0)
    heap = torch.stack([sample.heap_vectors for sample in samples], dim=0)
    teacher = torch.stack([sample.teacher_scores for sample in samples], dim=0)
    padding_mask = torch.ones(evicted.shape[:2], dtype=torch.bool)
    return {
        "backbones": backbones,
        "evicted_vectors": evicted,
        "heap_vectors": heap,
        "teacher_scores": teacher,
        "padding_mask": padding_mask,
    }
