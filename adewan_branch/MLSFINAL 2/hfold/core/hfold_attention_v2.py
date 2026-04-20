"""
HFOLD Attention Implementation v2

Maintains per-head heaps of bounded size s; each step inserts top-q keys by
pre-softmax scores, pops top-e entries for joint attention with the sliding
window, then folds removed key information into remaining heap entries.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeapEntry:
    """Single entry in the HFOLD heap (min-heap on attention_score keeps top-s by magnitude)."""

    def __init__(self, attention_score: float, token_idx: int, heap_idx: int):
        self.attention_score = attention_score
        self.token_idx = token_idx
        self.heap_idx = heap_idx
        self.folded_state: Optional[torch.Tensor] = None

    def __lt__(self, other: HeapEntry) -> bool:
        if self.attention_score != other.attention_score:
            return self.attention_score < other.attention_score
        return self.heap_idx < other.heap_idx

    def __repr__(self) -> str:
        return f"HeapEntry(score={self.attention_score:.3f}, pos={self.token_idx})"


@dataclass
class HeapHeadBucket:
    """Per-(batch, head) heap container with stable uid generation and sliding-window tracking."""

    heap: List[HeapEntry] = field(default_factory=list)
    next_uid: int = 0
    prev_window_indices: List[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.heap)


def as_heap_bucket(h: Union[HeapHeadBucket, List]) -> HeapHeadBucket:
    """Normalize legacy bare list heaps to HeapHeadBucket."""
    if isinstance(h, HeapHeadBucket):
        return h
    b = HeapHeadBucket()
    b.heap = list(h)
    heapq.heapify(b.heap)
    return b


def copy_heap_bucket(bucket: HeapHeadBucket) -> HeapHeadBucket:
    """Shallow copy of heap list (entries shared)."""
    out = HeapHeadBucket()
    out.heap = list(bucket.heap)
    out.next_uid = bucket.next_uid
    out.prev_window_indices = list(bucket.prev_window_indices)
    return out


def copy_heap_bucket_deep(bucket: HeapHeadBucket) -> HeapHeadBucket:
    """Deep copy heap entries (folded_state cloned) for independent mutable state."""
    out = HeapHeadBucket()
    out.next_uid = bucket.next_uid
    out.prev_window_indices = list(bucket.prev_window_indices)
    for e in bucket.heap:
        ne = HeapEntry(e.attention_score, e.token_idx, e.heap_idx)
        ne.folded_state = e.folded_state.clone() if e.folded_state is not None else None
        out.heap.append(ne)
    heapq.heapify(out.heap)
    return out


class HFoldAttentionV2(nn.Module):
    """
    Single HFOLD attention step over one query position.

    Algorithm per (batch, head):
    1. Track keys that fell off the sliding window since last step (falloff).
    2. Compute pre-softmax scores for the current window; insert top-q entries into heap (size s).
       Record keys evicted by heapreplace as removed.
    3. Pop the e largest-score entries from the heap (not heappop).
    4. Single softmax over concatenated [window logits | heap logits]; values are window V and
       W_heap_v(V_idx) for heap slots. Heap keys use W_heap_k(K_idx + fold_proj(folded_state)).
    5. Fold a summary of removed vectors (falloff + evictions + popped) into each remaining heap entry.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int,
        heap_size: int,
        q_topk: int,
        e_pop: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.window_size = window_size
        self.heap_size = heap_size
        self.q_topk = q_topk
        self.e_pop = min(e_pop, heap_size)

        self.W_heap_k = nn.Linear(self.d_k, self.d_k)
        self.W_heap_v = nn.Linear(self.d_k, self.d_k)
        self.fold_combine = nn.Sequential(
            nn.Linear(self.d_k, self.d_k),
            nn.GELU(),
            nn.Linear(self.d_k, self.d_k),
        )
        self.fold_gate = nn.Linear(self.d_k, self.d_k)
        self.fold_state_proj = nn.Linear(self.d_k, self.d_k)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        current_token_idx: int,
        heaps: Optional[List[List[Union[HeapHeadBucket, List]]]] = None,
    ) -> Tuple[torch.Tensor, List[List[HeapHeadBucket]], Dict]:
        batch_size, n_heads, _, d_k = query.shape
        assert d_k == self.d_k

        if heaps is None:
            heaps = [[HeapHeadBucket() for _ in range(n_heads)] for _ in range(batch_size)]
        else:
            for b in range(batch_size):
                for h in range(n_heads):
                    heaps[b][h] = as_heap_bucket(heaps[b][h])

        debug_dict: Dict = {"avg_heap_size": 0.0, "keys_added": 0, "keys_retrieved": 0}
        outputs: List[torch.Tensor] = []
        total_heap_size = 0

        scale = 1.0 / math.sqrt(d_k)

        for b in range(batch_size):
            batch_heads: List[torch.Tensor] = []
            for h in range(n_heads):
                bucket = heaps[b][h]
                heap: List[HeapEntry] = bucket.heap

                q = query[b, h, :, :]  # (1, d_k)
                window_start = max(0, current_token_idx - self.window_size)
                cur_window_indices = list(range(window_start, current_token_idx + 1))

                # Keys that slid out of the window since last step
                removed_from_window: List[torch.Tensor] = []
                if bucket.prev_window_indices:
                    prev_set = set(bucket.prev_window_indices)
                    cur_set = set(cur_window_indices)
                    for gi in prev_set - cur_set:
                        if 0 <= gi <= current_token_idx:
                            removed_from_window.append(keys[b, h, gi, :].clone())
                bucket.prev_window_indices = cur_window_indices

                k_window = keys[b, h, window_start : current_token_idx + 1, :]
                v_window = values[b, h, window_start : current_token_idx + 1, :]
                lw = k_window.shape[0]

                scores_window = torch.matmul(q, k_window.transpose(0, 1)) * scale  # (1, lw)
                flat_scores = scores_window.view(-1)

                # Top-q insertions
                q_take = min(self.q_topk, flat_scores.numel())
                topk_scores, topk_indices = torch.topk(flat_scores, k=q_take)
                evicted_vectors: List[torch.Tensor] = []

                for i in range(q_take):
                    score = float(topk_scores[i].item())
                    wi = int(topk_indices[i].item())
                    global_idx = window_start + wi
                    uid = bucket.next_uid
                    bucket.next_uid += 1
                    entry = HeapEntry(score, global_idx, uid)

                    if len(heap) < self.heap_size:
                        heapq.heappush(heap, entry)
                        debug_dict["keys_added"] += 1
                    elif score > heap[0].attention_score:
                        old = heapq.heapreplace(heap, entry)
                        debug_dict["keys_added"] += 1
                        evicted_vectors.append(keys[b, h, old.token_idx, :].clone())

                # Pop e largest scores (exact removal)
                e_take = min(self.e_pop, len(heap))
                popped_entries: List[HeapEntry] = []
                if e_take > 0:
                    popped_entries = heapq.nlargest(e_take, heap)
                    popped_ids = {id(e) for e in popped_entries}
                    heap[:] = [e for e in heap if id(e) not in popped_ids]
                    heapq.heapify(heap)
                    debug_dict["keys_retrieved"] += len(popped_entries)

                # Build joint pre-softmax logits and values
                parts_scores: List[torch.Tensor] = []
                parts_values: List[torch.Tensor] = []

                parts_scores.append(scores_window.squeeze(0))
                parts_values.append(v_window)

                for ent in popped_entries:
                    base_k = keys[b, h, ent.token_idx, :]
                    if ent.folded_state is not None:
                        base_k = base_k + self.fold_state_proj(ent.folded_state)
                    k_h = self.W_heap_k(base_k.unsqueeze(0)).squeeze(0)
                    v_h_raw = values[b, h, ent.token_idx, :]
                    v_h = self.W_heap_v(v_h_raw.unsqueeze(0)).squeeze(0)
                    s_h = (q.squeeze(0) * k_h).sum() * scale
                    parts_scores.append(s_h.unsqueeze(0))
                    parts_values.append(v_h.unsqueeze(0))

                cat_scores = torch.cat(parts_scores, dim=0)
                cat_values = torch.cat(parts_values, dim=0)
                attn_w = F.softmax(cat_scores, dim=0)
                attn_w = self.dropout(attn_w)
                output = attn_w.unsqueeze(0) @ cat_values  # (1, d_k)

                # Folding: aggregate removed tokens (window falloff, heap evictions, popped)
                removed_vecs: List[torch.Tensor] = list(removed_from_window)
                removed_vecs.extend(evicted_vectors)
                for ent in popped_entries:
                    removed_vecs.append(keys[b, h, ent.token_idx, :].clone())

                if removed_vecs and len(heap) > 0:
                    stacked = torch.stack(removed_vecs, dim=0)
                    removed_summary = stacked.mean(dim=0, keepdim=True)
                    folded_delta = self.fold_combine(removed_summary).squeeze(0)
                    gate_in = folded_delta
                    gate = torch.sigmoid(self.fold_gate(gate_in))
                    for entry in heap:
                        if entry.folded_state is None:
                            entry.folded_state = gate * folded_delta
                        else:
                            entry.folded_state = (1 - gate) * entry.folded_state + gate * folded_delta

                batch_heads.append(output)
                total_heap_size += len(heap)

            outputs.append(torch.stack(batch_heads, dim=0))

        output_stacked = torch.stack(outputs, dim=0)
        denom = batch_size * n_heads
        debug_dict["avg_heap_size"] = total_heap_size / denom if denom else 0.0
        return output_stacked, heaps, debug_dict


class HFoldMultiHeadAttention(nn.Module):
    """Multi-head HFOLD attention with Q, K, V projections."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int,
        heap_size: int,
        q_topk: int,
        e_pop: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.hfold = HFoldAttentionV2(
            d_model=d_model,
            n_heads=n_heads,
            window_size=window_size,
            heap_size=heap_size,
            q_topk=q_topk,
            e_pop=e_pop,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        current_token_idx: int,
        heaps: Optional[List[List]] = None,
    ) -> Tuple[torch.Tensor, Optional[List[List]], Dict]:
        batch_size, seq_len, _ = x.shape

        q = self.W_q(x[:, -1:, :])
        k = self.W_k(x)
        v = self.W_v(x)

        q = q.view(batch_size, 1, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        attn_output, heaps, debug_info = self.hfold(q, k, v, current_token_idx, heaps)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, 1, self.d_model)
        output = self.W_o(attn_output)
        output = self.dropout(output)

        return output, heaps, debug_info
