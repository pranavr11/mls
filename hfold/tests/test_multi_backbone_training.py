"""Cross-backbone training must work even when raw hidden sizes differ.

This test guards the most important fix in this audit pass: the collate
function and trainers tolerate per-sample variable-shape `heap_vectors` and
`evicted_vectors` (Pythia 256 vs GPT-2 768 in production), and only stack
after each row is mapped to the shared adapter dim.
"""
from __future__ import annotations

import os

import torch

from hfold.config.schema import HFoldConfig, HFoldModelConfig, HFoldTrainingConfig
from hfold.data.hidden_state_dataset import HiddenStateShardDataset
from hfold.training.train_embedding import train_embedding_model
from hfold.training.train_relevancy import train_relevancy_model


def _shard_with_dim(path: str, backbone: str, hidden_size: int, max_heap: int, count: int) -> None:
    payload = []
    gen = torch.Generator().manual_seed(hash(backbone) & 0xFFFF)
    for _ in range(count):
        scores = torch.softmax(torch.rand(max_heap, generator=gen), dim=0)
        payload.append(
            {
                "backbone": backbone,
                "heap_vectors": torch.randn(max_heap, hidden_size, generator=gen),
                "evicted_vectors": torch.randn(max_heap, hidden_size, generator=gen),
                "teacher_scores": scores,
            }
        )
    torch.save(payload, path)


def test_trainers_handle_heterogeneous_backbone_dims(tmp_path):
    pythia_dir = tmp_path / "pythia"
    gpt2_dir = tmp_path / "gpt2"
    os.makedirs(pythia_dir)
    os.makedirs(gpt2_dir)
    _shard_with_dim(str(pythia_dir / "shard_0000.pt"), backbone="pythia", hidden_size=12, max_heap=4, count=4)
    _shard_with_dim(str(gpt2_dir / "shard_0000.pt"), backbone="gpt2", hidden_size=20, max_heap=4, count=4)

    dataset = HiddenStateShardDataset([str(pythia_dir), str(gpt2_dir)])
    config = HFoldConfig(
        model=HFoldModelConfig(hidden_size=20, num_heads=4, max_heap_size=4, top_w=4, pop_k=4, adapter_dim=16),
        training=HFoldTrainingConfig(num_epochs=1, max_steps=2, batch_size=3),
    )

    emb = train_embedding_model(
        config=config,
        dataset=dataset,
        backbone_dims={"pythia": 12, "gpt2": 20},
    )
    rel = train_relevancy_model(
        config=config,
        dataset=dataset,
        embedding_model=emb.model,
        adapters=emb.adapters,
    )
    assert emb.final_loss >= 0.0
    assert rel.final_loss >= 0.0
