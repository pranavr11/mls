import torch

from hfold.config.schema import HFoldConfig, HFoldModelConfig
from hfold.inference.hfold_runtime import HFoldRuntime


class DummyEmbedding:
    def encode_summary(self, vectors: torch.Tensor, padding_mask=None) -> torch.Tensor:
        del padding_mask
        return vectors.mean(dim=1)


class DummyRelevancy:
    def score_heap(self, summary: torch.Tensor, heap_vectors: torch.Tensor) -> torch.Tensor:
        del summary
        return torch.ones(heap_vectors.size(0), heap_vectors.size(1), device=heap_vectors.device)


def test_fold_matches_h_plus_r_times_g():
    runtime = HFoldRuntime(HFoldConfig(model=HFoldModelConfig(hidden_size=4, num_heads=2, max_heap_size=2, top_w=2, pop_k=1)))
    runtime.prime_timestep_zero(
        layer_index=0,
        vectors=torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0]]),
        scores=torch.tensor([0.9, 0.8]),
        token_positions=torch.tensor([0, 1]),
        head_indices=torch.tensor([0, 0]),
        time_index=0,
    )
    popped = runtime.pop_top_k(layer_index=0)
    before = [entry.vector.clone() for entry in runtime.state.layers[0].heap]
    runtime.step_with_reinsert_and_fold(
        layer_index=0,
        popped_entries=popped,
        transformed_popped_vectors=torch.tensor([[2.0, 0.0, 0.0, 0.0]]),
        new_vectors=torch.tensor([[3.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0]]),
        new_scores=torch.tensor([0.2, 0.1]),
        new_token_positions=torch.tensor([2, 3]),
        new_head_indices=torch.tensor([0, 0]),
        time_index=1,
        embedding_model=DummyEmbedding(),
        relevancy_model=DummyRelevancy(),
    )
    after = [entry.vector for entry in runtime.state.layers[0].heap]
    assert len(after) == 2
    assert any(not torch.allclose(before[i], after[i]) for i in range(min(len(before), len(after))))
