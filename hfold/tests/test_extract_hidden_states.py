"""Tests for the real hidden-state extractor.

We use a tiny in-process model that emits well-defined attention/hidden states
so we can verify shape and label-distribution invariants without downloading
a HuggingFace checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from hfold.data.extract_hidden_states import (
    ExtractionConfig,
    extract_one_chunk,
    extract_to_shards,
)
from hfold.data.hidden_state_dataset import HiddenStateShardDataset


@dataclass
class _Output:
    hidden_states: tuple[torch.Tensor, ...]
    attentions: tuple[torch.Tensor, ...]


class _DeterministicHFLikeModel(nn.Module):
    """Minimal HF-shaped model: returns one hidden-states layer and one attention."""

    def __init__(self, hidden_size: int = 16, num_heads: int = 4) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.embed = nn.Embedding(64, hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids, attention_mask=None, output_attentions=False, output_hidden_states=False, use_cache=False, return_dict=True, **_):
        del attention_mask, use_cache, return_dict
        h = self.embed(input_ids)
        h = self.proj(h)
        b, s, _ = h.shape
        # build a deterministic attention map: each query attends mostly to
        # its immediate predecessor → top heap_idx will favor recent tokens.
        attn = torch.zeros(b, self.num_heads, s, s)
        for q in range(s):
            for k in range(q + 1):
                attn[:, :, q, k] = 1.0 / (q - k + 1)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return _Output(
            hidden_states=(h,) if output_hidden_states else (),
            attentions=(attn,) if output_attentions else (),
        )


def test_extract_one_chunk_returns_valid_tuples():
    torch.manual_seed(0)
    model = _DeterministicHFLikeModel(hidden_size=16, num_heads=4)
    config = ExtractionConfig(
        backbone="pythia",
        chunk_len=64,
        max_heap_size=4,
        num_anchors_per_chunk=2,
        min_anchor_position=10,
        seed=0,
    )
    input_ids = torch.randint(0, 64, (1, 64), generator=torch.Generator().manual_seed(0))
    samples = extract_one_chunk(
        model=model,
        input_ids=input_ids,
        attention_mask=None,
        config=config,
        generator=torch.Generator(device="cpu").manual_seed(0),
    )
    assert 0 < len(samples) <= 2
    for s in samples:
        assert s["backbone"] == "pythia"
        assert s["heap_vectors"].shape == (4, 16)
        assert s["evicted_vectors"].shape == (4, 16)
        assert s["teacher_scores"].shape == (4,)
        assert torch.all(s["teacher_scores"] >= 0)
        assert abs(float(s["teacher_scores"].sum().item()) - 1.0) < 1e-5


def test_extract_to_shards_writes_and_dataset_reads(tmp_path):
    torch.manual_seed(0)
    model = _DeterministicHFLikeModel(hidden_size=16, num_heads=4)
    config = ExtractionConfig(
        backbone="pythia",
        chunk_len=32,
        max_heap_size=4,
        num_anchors_per_chunk=2,
        min_anchor_position=10,
        seed=0,
    )

    def _loader():
        for _ in range(4):
            yield {"input_ids": torch.randint(0, 64, (1, 32))}

    output_dir = str(tmp_path)
    total = extract_to_shards(
        model=model,
        dataloader=_loader(),
        output_dir=output_dir,
        config=config,
        samples_per_shard=3,
        max_chunks=4,
    )
    assert total > 0

    dataset = HiddenStateShardDataset([output_dir])
    assert len(dataset) == total
    sample = dataset[0]
    assert sample.heap_vectors.shape == (4, 16)
    assert sample.evicted_vectors.shape == (4, 16)
    assert sample.teacher_scores.shape == (4,)
