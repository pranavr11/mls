import torch

from hfold.config.schema import HFoldConfig, HFoldModelConfig
from hfold.inference.hfold_runtime import HFoldRuntime


def test_pop_k_zero_disables_retrieval():
    runtime = HFoldRuntime(HFoldConfig(model=HFoldModelConfig(hidden_size=8, num_heads=2, max_heap_size=4, pop_k=0)))
    runtime.prime_timestep_zero(
        layer_index=0,
        vectors=torch.randn(4, 8),
        scores=torch.rand(4),
        token_positions=torch.arange(4),
        head_indices=torch.zeros(4, dtype=torch.long),
        time_index=0,
    )
    popped = runtime.pop_top_k(layer_index=0)
    assert popped == []


def test_heap_size_zero_disables_storage():
    runtime = HFoldRuntime(HFoldConfig(model=HFoldModelConfig(hidden_size=8, num_heads=2, max_heap_size=0, top_w=4)))
    artifacts = runtime.prime_timestep_zero(
        layer_index=0,
        vectors=torch.randn(4, 8),
        scores=torch.rand(4),
        token_positions=torch.arange(4),
        head_indices=torch.zeros(4, dtype=torch.long),
        time_index=0,
    )
    assert len(runtime.state.layers[0].heap) == 0
    assert len(artifacts.evicted_entries) == 0
