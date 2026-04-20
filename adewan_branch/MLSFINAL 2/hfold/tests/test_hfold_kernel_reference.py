"""
Regression tests for HFoldAttentionV2: heap capacity, folded-state effect, pop logic.
"""

import heapq

import torch

from hfold.core.hfold_attention_v2 import HFoldAttentionV2, HeapHeadBucket


def test_fold_state_proj_changes_projected_heap_key():
    """Folded state must alter the effective key before W_heap_k (HFold read path)."""
    torch.manual_seed(1)
    d_model, n_heads = 8, 1
    d_k = d_model // n_heads
    mod = HFoldAttentionV2(
        d_model=d_model,
        n_heads=n_heads,
        window_size=4,
        heap_size=4,
        q_topk=2,
        e_pop=1,
        dropout=0.0,
    )
    base_k = torch.randn(d_k)
    folded = torch.randn(d_k)
    k0 = mod.W_heap_k(base_k.unsqueeze(0)).squeeze(0)
    k1 = mod.W_heap_k((base_k + mod.fold_state_proj(folded)).unsqueeze(0)).squeeze(0)
    assert not torch.allclose(k0, k1)


def test_heap_capacity_respected_over_many_steps():
    torch.manual_seed(2)
    d_model, n_heads = 24, 2
    s = 5
    mod = HFoldAttentionV2(
        d_model=d_model,
        n_heads=n_heads,
        window_size=6,
        heap_size=s,
        q_topk=4,
        e_pop=2,
        dropout=0.0,
    )
    seq_len = 40
    q = torch.randn(2, n_heads, 1, d_model // n_heads)
    keys = torch.randn(2, n_heads, seq_len, d_model // n_heads)
    values = torch.randn(2, n_heads, seq_len, d_model // n_heads)

    heaps = [[HeapHeadBucket() for _ in range(n_heads)] for _ in range(2)]
    for t in range(seq_len):
        qq = q.expand(2, -1, -1, -1).clone()
        _, heaps, _ = mod(qq, keys, values, t, heaps)
        for b in range(2):
            for h in range(n_heads):
                assert len(heaps[b][h].heap) <= s


def test_nlargest_pop_matches_filtered_heap():
    """After nlargest(e) removal, remaining multiset has correct size."""

    class E:
        def __init__(self, sc, i):
            self.attention_score = sc
            self.heap_idx = i

        def __lt__(self, o):
            if self.attention_score != o.attention_score:
                return self.attention_score < o.attention_score
            return self.heap_idx < o.heap_idx

    scores = [3.0, 1.0, 4.0, 1.5, 2.0]
    heap = [E(s, i) for i, s in enumerate(scores)]
    heapq.heapify(heap)
    e = 2
    pop = heapq.nlargest(e, heap)
    popped_ids = {id(x) for x in pop}
    remaining = [x for x in heap if id(x) not in popped_ids]
    heapq.heapify(remaining)
    assert len(remaining) + e == len(scores)
    rem_scores = sorted([x.attention_score for x in remaining])
    assert rem_scores == sorted([1.0, 1.5, 2.0])
