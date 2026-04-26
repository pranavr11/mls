import torch

from hfold.inference.heap_state import HFoldHeapEntry
from hfold.inference.priority_heap import BoundedMaxHeap


def _entry(score: float, entry_id: int) -> HFoldHeapEntry:
    return HFoldHeapEntry(
        score=score,
        vector=torch.tensor([score]),
        token_position=entry_id,
        layer_index=0,
        head_index=0,
        time_index=0,
        id=entry_id,
    )


def test_bounded_heap_retains_highest_scores():
    heap = BoundedMaxHeap(capacity=3)
    evicted = heap.push_many([_entry(0.1, 0), _entry(0.9, 1), _entry(0.3, 2), _entry(0.7, 3)])
    assert len(heap) == 3
    kept_scores = [entry.score for entry in heap.peek_all()]
    assert kept_scores == [0.9, 0.7, 0.3]
    assert [entry.score for entry in evicted] == [0.1]


def test_pop_top_k_tie_breaks_by_entry_id():
    heap = BoundedMaxHeap(capacity=5)
    heap.push_many([_entry(0.8, 1), _entry(0.8, 0), _entry(0.4, 3)])
    popped = heap.pop_top_k(2)
    # Deterministic contract for this heap implementation: ties are broken by
    # ascending entry id.
    assert [(entry.score, entry.id) for entry in popped] == [(0.8, 0), (0.8, 1)]
    remaining = heap.peek_all()
    assert len(remaining) == 1
    assert remaining[0].score == 0.4
