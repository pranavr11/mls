import torch

from hfold.config.schema import HFoldConfig, HFoldModelConfig
from hfold.inference.hfold_runtime import HFoldRuntime


def _run_once(seed: int):
    torch.manual_seed(seed)
    runtime = HFoldRuntime(HFoldConfig(model=HFoldModelConfig(hidden_size=8, num_heads=2, max_heap_size=4)))
    vectors = torch.randn(4, 8)
    scores = torch.randn(4).softmax(dim=0)
    runtime.prime_timestep_zero(
        layer_index=0,
        vectors=vectors,
        scores=scores,
        token_positions=torch.arange(4),
        head_indices=torch.zeros(4, dtype=torch.long),
        time_index=0,
    )
    return [entry.score for entry in runtime.state.layers[0].heap]


def test_heap_behavior_deterministic_under_seed():
    a = _run_once(123)
    b = _run_once(123)
    assert a == b
