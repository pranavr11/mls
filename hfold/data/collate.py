from __future__ import annotations

import torch

from .hidden_state_dataset import HiddenStateSample


def collate_hfold_samples(samples: list[HiddenStateSample]) -> dict:
    """Collate a batch of HiddenStateSamples for embedding/relevancy training.

    Across backbones, raw `hidden_size` may differ (Pythia: 256, GPT-2: 768),
    so stacking raw vectors into one tensor is not always valid. We instead
    return per-sample lists of variable-shape tensors. The trainer applies
    each sample's backbone adapter individually and only stacks **after**
    everything is in the shared latent dim.
    """
    backbones = [sample.backbone for sample in samples]
    evicted = [sample.evicted_vectors for sample in samples]
    heap = [sample.heap_vectors for sample in samples]
    # teacher_scores is always shape [max_heap_size]; safe to stack.
    teacher = torch.stack([sample.teacher_scores for sample in samples], dim=0)
    padding_mask = torch.ones(teacher.shape, dtype=torch.bool)
    return {
        "backbones": backbones,
        "evicted_vectors": evicted,
        "heap_vectors": heap,
        "teacher_scores": teacher,
        "padding_mask": padding_mask,
    }
