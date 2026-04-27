import torch

from hfold.inference.tensor_heap import HFoldTensorBundle, pop_top_k_tensor, push_many_tensor


def _bundle(scores: list[float], hidden_size: int = 4) -> HFoldTensorBundle:
    n = len(scores)
    dev = torch.device("cpu")
    return HFoldTensorBundle(
        scores=torch.tensor(scores, dtype=torch.float32, device=dev),
        vectors=torch.arange(n * hidden_size, dtype=torch.float32, device=dev).view(n, hidden_size),
        token_positions=torch.arange(n, dtype=torch.long, device=dev),
        head_indices=torch.zeros(n, dtype=torch.long, device=dev),
        time_indices=torch.zeros(n, dtype=torch.long, device=dev),
        entry_ids=torch.arange(n, dtype=torch.long, device=dev),
    )


def test_push_many_tensor_respects_capacity_and_returns_evicted():
    heap = HFoldTensorBundle.empty(hidden_size=4)
    candidates = _bundle([0.1, 0.9, 0.3, 0.7], hidden_size=4)
    heap, evicted = push_many_tensor(heap=heap, candidates=candidates, capacity=3)

    assert len(heap) == 3
    assert len(evicted) == 1
    assert torch.allclose(heap.scores, torch.tensor([0.9, 0.7, 0.3]))
    assert torch.allclose(evicted.scores, torch.tensor([0.1]))


def test_pop_top_k_tensor_returns_top_scores_and_keeps_remainder():
    heap = _bundle([0.2, 0.8, 0.6, 0.1], hidden_size=4)
    remaining, popped = pop_top_k_tensor(heap=heap, k=2)

    assert len(popped) == 2
    assert torch.allclose(popped.scores, torch.tensor([0.8, 0.6]))
    assert len(remaining) == 2
    remaining_scores = sorted(float(x.item()) for x in remaining.scores)
    assert abs(remaining_scores[0] - 0.1) <= 1e-6
    assert abs(remaining_scores[1] - 0.2) <= 1e-6


def test_equal_score_behavior_uses_score_multiset_contract():
    heap = HFoldTensorBundle.empty(hidden_size=4)
    candidates = _bundle([0.5, 0.5, 0.5, 0.2], hidden_size=4)
    heap, evicted = push_many_tensor(heap=heap, candidates=candidates, capacity=3)

    # Tie-order may vary by backend/invocation; assert score multiset only.
    kept_scores = sorted(float(x.item()) for x in heap.scores)
    evicted_scores = sorted(float(x.item()) for x in evicted.scores)
    assert all(abs(score - 0.5) <= 1e-6 for score in kept_scores)
    assert len(evicted_scores) == 1
    assert abs(evicted_scores[0] - 0.2) <= 1e-6
