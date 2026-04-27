from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class HFoldTensorBundle:
    """
    Struct-of-arrays heap representation.

    All fields share the same leading dimension N.
    """

    scores: torch.Tensor
    vectors: torch.Tensor
    token_positions: torch.Tensor
    head_indices: torch.Tensor
    time_indices: torch.Tensor
    entry_ids: torch.Tensor

    def __len__(self) -> int:
        return int(self.scores.numel())

    @property
    def hidden_size(self) -> int:
        if self.vectors.dim() != 2:
            return 0
        return int(self.vectors.size(-1))

    @staticmethod
    def empty(
        *,
        hidden_size: int,
        device: torch.device | None = None,
        vector_dtype: torch.dtype = torch.float32,
        score_dtype: torch.dtype = torch.float32,
    ) -> "HFoldTensorBundle":
        dev = device if device is not None else torch.device("cpu")
        return HFoldTensorBundle(
            scores=torch.empty((0,), device=dev, dtype=score_dtype),
            vectors=torch.empty((0, hidden_size), device=dev, dtype=vector_dtype),
            token_positions=torch.empty((0,), device=dev, dtype=torch.long),
            head_indices=torch.empty((0,), device=dev, dtype=torch.long),
            time_indices=torch.empty((0,), device=dev, dtype=torch.long),
            entry_ids=torch.empty((0,), device=dev, dtype=torch.long),
        )

    def to(self, *, device: torch.device, vector_dtype: torch.dtype, score_dtype: torch.dtype) -> "HFoldTensorBundle":
        return HFoldTensorBundle(
            scores=self.scores.to(device=device, dtype=score_dtype),
            vectors=self.vectors.to(device=device, dtype=vector_dtype),
            token_positions=self.token_positions.to(device=device),
            head_indices=self.head_indices.to(device=device),
            time_indices=self.time_indices.to(device=device),
            entry_ids=self.entry_ids.to(device=device),
        )


def _normalize_empty_for_like(
    bundle: HFoldTensorBundle,
    *,
    like_vectors: torch.Tensor,
    like_scores: torch.Tensor,
) -> HFoldTensorBundle:
    if len(bundle) > 0:
        return bundle.to(
            device=like_vectors.device,
            vector_dtype=like_vectors.dtype,
            score_dtype=like_scores.dtype,
        )
    return HFoldTensorBundle.empty(
        hidden_size=int(like_vectors.size(-1)),
        device=like_vectors.device,
        vector_dtype=like_vectors.dtype,
        score_dtype=like_scores.dtype,
    )


def _take_rows(bundle: HFoldTensorBundle, indices: torch.Tensor) -> HFoldTensorBundle:
    if indices.numel() == 0:
        return HFoldTensorBundle.empty(
            hidden_size=bundle.hidden_size,
            device=bundle.vectors.device,
            vector_dtype=bundle.vectors.dtype,
            score_dtype=bundle.scores.dtype,
        )
    return HFoldTensorBundle(
        scores=torch.take_along_dim(bundle.scores, indices, dim=0),
        vectors=bundle.vectors.index_select(0, indices),
        token_positions=torch.take_along_dim(bundle.token_positions, indices, dim=0),
        head_indices=torch.take_along_dim(bundle.head_indices, indices, dim=0),
        time_indices=torch.take_along_dim(bundle.time_indices, indices, dim=0),
        entry_ids=torch.take_along_dim(bundle.entry_ids, indices, dim=0),
    )


def _cat_rows(a: HFoldTensorBundle, b: HFoldTensorBundle) -> HFoldTensorBundle:
    if len(a) == 0:
        return b
    if len(b) == 0:
        return a
    return HFoldTensorBundle(
        scores=torch.cat([a.scores, b.scores], dim=0),
        vectors=torch.cat([a.vectors, b.vectors], dim=0),
        token_positions=torch.cat([a.token_positions, b.token_positions], dim=0),
        head_indices=torch.cat([a.head_indices, b.head_indices], dim=0),
        time_indices=torch.cat([a.time_indices, b.time_indices], dim=0),
        entry_ids=torch.cat([a.entry_ids, b.entry_ids], dim=0),
    )


def push_many_tensor(
    *,
    heap: HFoldTensorBundle,
    candidates: HFoldTensorBundle,
    capacity: int,
) -> tuple[HFoldTensorBundle, HFoldTensorBundle]:
    """
    Merge heap + candidates and keep best `capacity` entries by score.
    Ties intentionally follow torch.topk behavior (non-stable).
    """

    if len(candidates) == 0:
        return heap, HFoldTensorBundle.empty(
            hidden_size=heap.hidden_size,
            device=heap.vectors.device,
            vector_dtype=heap.vectors.dtype,
            score_dtype=heap.scores.dtype,
        )

    heap_aligned = _normalize_empty_for_like(heap, like_vectors=candidates.vectors, like_scores=candidates.scores)
    merged = _cat_rows(heap_aligned, candidates)
    total = len(merged)
    keep_n = min(max(capacity, 0), total)

    if keep_n <= 0:
        empty = HFoldTensorBundle.empty(
            hidden_size=merged.hidden_size,
            device=merged.vectors.device,
            vector_dtype=merged.vectors.dtype,
            score_dtype=merged.scores.dtype,
        )
        return empty, merged

    _, keep_idx = torch.topk(merged.scores, k=keep_n, largest=True, sorted=True)
    keep = _take_rows(merged, keep_idx)

    if keep_n >= total:
        evicted = HFoldTensorBundle.empty(
            hidden_size=merged.hidden_size,
            device=merged.vectors.device,
            vector_dtype=merged.vectors.dtype,
            score_dtype=merged.scores.dtype,
        )
        return keep, evicted

    all_idx = torch.arange(total, device=merged.scores.device, dtype=torch.long)
    keep_mask = torch.zeros(total, device=merged.scores.device, dtype=torch.bool)
    keep_mask[keep_idx] = True
    evict_idx = all_idx[~keep_mask]
    evicted = _take_rows(merged, evict_idx)
    return keep, evicted


def pop_top_k_tensor(
    *,
    heap: HFoldTensorBundle,
    k: int,
) -> tuple[HFoldTensorBundle, HFoldTensorBundle]:
    if k <= 0 or len(heap) == 0:
        empty = HFoldTensorBundle.empty(
            hidden_size=heap.hidden_size,
            device=heap.vectors.device,
            vector_dtype=heap.vectors.dtype,
            score_dtype=heap.scores.dtype,
        )
        return heap, empty

    pop_n = min(int(k), len(heap))
    _, pop_idx = torch.topk(heap.scores, k=pop_n, largest=True, sorted=True)
    popped = _take_rows(heap, pop_idx)

    all_idx = torch.arange(len(heap), device=heap.scores.device, dtype=torch.long)
    keep_mask = torch.zeros(len(heap), device=heap.scores.device, dtype=torch.bool)
    keep_mask[pop_idx] = True
    rem_idx = all_idx[~keep_mask]
    remaining = _take_rows(heap, rem_idx)
    return remaining, popped

