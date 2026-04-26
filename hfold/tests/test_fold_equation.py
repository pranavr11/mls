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
    """Verify h_i := h_i + r_i * g exactly when adapters are identity-like and
    the relevancy model returns a deterministic score per heap entry.
    """

    class IdentityEmbedding:
        def encode_summary(self, vectors: torch.Tensor, padding_mask=None) -> torch.Tensor:
            del padding_mask
            return vectors.mean(dim=1)

    class FixedRelevancy:
        def __init__(self, score: float) -> None:
            self.score = score

        def score_heap(self, summary: torch.Tensor, heap_vectors: torch.Tensor) -> torch.Tensor:
            del summary
            return torch.full((heap_vectors.size(0), heap_vectors.size(1)), self.score)

    # Heap capacity 2, top_w 2, pop_k 1 ensures exactly one eviction.
    runtime = HFoldRuntime(HFoldConfig(model=HFoldModelConfig(hidden_size=4, num_heads=2, max_heap_size=2, top_w=2, pop_k=1)))
    initial_vectors = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
        ]
    )
    runtime.prime_timestep_zero(
        layer_index=0,
        vectors=initial_vectors,
        scores=torch.tensor([0.9, 0.8]),
        token_positions=torch.tensor([0, 1]),
        head_indices=torch.tensor([0, 0]),
        time_index=0,
    )
    popped = runtime.pop_top_k(layer_index=0)
    heap_before = [
        (entry.token_position, entry.vector.clone())
        for entry in runtime.state.layers[0].heap
    ]

    new_vectors = torch.tensor(
        [
            [3.0, 0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0, 0.0],
        ]
    )
    transformed_popped = torch.tensor([[2.0, 0.0, 0.0, 0.0]])

    fixed_score = 0.5
    artifacts = runtime.step_with_reinsert_and_fold(
        layer_index=0,
        popped_entries=popped,
        transformed_popped_vectors=transformed_popped,
        new_vectors=new_vectors,
        new_scores=torch.tensor([0.2, 0.1]),
        new_token_positions=torch.tensor([2, 3]),
        new_head_indices=torch.tensor([0, 0]),
        time_index=1,
        embedding_model=IdentityEmbedding(),
        relevancy_model=FixedRelevancy(fixed_score),
    )

    assert artifacts.summary_embedding is not None
    g = artifacts.summary_embedding[0]
    after = runtime.state.layers[0].heap
    assert len(after) == 2

    # The heap is rebuilt by score-sorting; pair each kept entry with the right
    # pre-fold vector by token_position. Reconstruct the expected h + r * g.
    pre_fold_by_position = {pos: vec for pos, vec in heap_before}
    pre_fold_by_position[popped[0].token_position] = transformed_popped[0]
    new_positions_after_dedup = [int(t.item()) for t in torch.tensor([2, 3])]
    for vec, pos in zip(new_vectors, new_positions_after_dedup):
        if pos not in {entry.token_position for entry in popped}:
            pre_fold_by_position[pos] = vec

    for entry in after:
        original = pre_fold_by_position[entry.token_position]
        expected = original + fixed_score * g
        assert torch.allclose(entry.vector, expected, atol=1e-5), (
            f"fold mismatch at token_position={entry.token_position}: "
            f"got {entry.vector.tolist()} expected {expected.tolist()}"
        )
