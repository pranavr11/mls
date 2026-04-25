from __future__ import annotations

from heapq import heapify, heappop, heappush

from .heap_state import HFoldHeapEntry


class BoundedMaxHeap:
    """
    Deterministic bounded max-heap via negated min-heap keys.
    Tie-breaks use monotonically increasing entry id.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 0:
            raise ValueError("capacity must be >= 0")
        self.capacity = capacity
        self._heap: list[tuple[float, int, HFoldHeapEntry]] = []

    def __len__(self) -> int:
        return len(self._heap)

    def peek_all(self) -> list[HFoldHeapEntry]:
        data = sorted(self._heap, key=lambda x: (x[0], x[1]))
        return [item[2] for item in data]

    def pop_top_k(self, k: int) -> list[HFoldHeapEntry]:
        if k <= 0 or not self._heap:
            return []
        # materialize deterministic order by score desc then id asc
        sorted_entries = sorted(self._heap, key=lambda x: (x[0], x[1]))
        top = sorted_entries[:k]
        keep = sorted_entries[k:]
        self._heap = keep
        heapify(self._heap)
        return [item[2] for item in top]

    def push_many(self, entries: list[HFoldHeapEntry]) -> list[HFoldHeapEntry]:
        if not entries:
            return []
        evicted: list[HFoldHeapEntry] = []
        for entry in entries:
            wrapped = (-float(entry.score), entry.id, entry)
            heappush(self._heap, wrapped)
            if self.capacity == 0:
                _, _, removed = heappop(self._heap)
                evicted.append(removed)
                continue
            if len(self._heap) > self.capacity:
                # Remove lowest-scoring element among kept set:
                # easiest deterministic route is full sort + truncate.
                sorted_entries = sorted(self._heap, key=lambda x: (x[0], x[1]))
                kept = sorted_entries[: self.capacity]
                overflow = sorted_entries[self.capacity :]
                self._heap = kept
                heapify(self._heap)
                evicted.extend([item[2] for item in overflow])
        return evicted
