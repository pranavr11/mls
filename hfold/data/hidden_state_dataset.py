"""Datasets that supply training tuples to the embedding/relevancy trainers.

Two implementations live here:

  * `HiddenStateShardDataset` — the production dataset. Reads `.pt` shards
    written by `hfold.data.extract_hidden_states.extract_to_shards`. Each
    shard is a list of dicts with keys
    {backbone, heap_vectors, evicted_vectors, teacher_scores}.
  * `SyntheticHiddenStateDataset` — a unit-test fixture that produces
    deterministic random tensors. It is intentionally restricted to test
    code paths and is NOT used by the production training scripts.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class HiddenStateSample:
    backbone: str
    evicted_vectors: torch.Tensor
    heap_vectors: torch.Tensor
    teacher_scores: torch.Tensor


class HiddenStateShardDataset(Dataset):
    """Reads training tuples from one or more shard directories.

    Each shard file is a `.pt` list of dicts; we flatten across all matching
    files and serve them via `__getitem__`. Loading is lazy (mmap-style):
    we keep an index of (file_path, in_shard_index) per sample and load on
    access. This keeps memory bounded for large datasets.
    """

    def __init__(self, shard_dirs: list[str], glob_pattern: str = "shard_*.pt") -> None:
        if not shard_dirs:
            raise ValueError("HiddenStateShardDataset requires at least one shard directory.")
        self._files: list[str] = []
        for shard_dir in shard_dirs:
            matched = sorted(glob.glob(os.path.join(shard_dir, glob_pattern)))
            if not matched:
                raise ValueError(f"No shards matched '{glob_pattern}' in {shard_dir}")
            self._files.extend(matched)
        # Build index by reading shard sizes once; full payloads stay on disk.
        self._index: list[tuple[int, int]] = []
        for file_idx, path in enumerate(self._files):
            shard = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(shard, list):
                raise ValueError(f"Shard {path} must contain a list of dicts.")
            for entry_idx in range(len(shard)):
                self._index.append((file_idx, entry_idx))
        self._cache_path: str | None = None
        self._cache_payload: list[dict] | None = None

    def __len__(self) -> int:
        return len(self._index)

    def _load_shard(self, path: str) -> list[dict]:
        if self._cache_path == path and self._cache_payload is not None:
            return self._cache_payload
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self._cache_path = path
        self._cache_payload = payload
        return payload

    def __getitem__(self, index: int) -> HiddenStateSample:
        file_idx, entry_idx = self._index[index]
        shard = self._load_shard(self._files[file_idx])
        entry = shard[entry_idx]
        return HiddenStateSample(
            backbone=str(entry["backbone"]),
            evicted_vectors=entry["evicted_vectors"],
            heap_vectors=entry["heap_vectors"],
            teacher_scores=entry["teacher_scores"],
        )


class SyntheticHiddenStateDataset(Dataset):
    """Test-only synthetic fixture. Production code MUST use shards.

    This dataset is preserved exclusively so unit tests can exercise the
    training/inference plumbing without depending on a real backbone forward
    pass. It is forbidden in production scripts and CLIs.
    """

    def __init__(
        self,
        *,
        size: int,
        backbone: str,
        hidden_size: int,
        max_heap_size: int,
        seed: int = 0,
    ) -> None:
        self.size = size
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.max_heap_size = max_heap_size
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> HiddenStateSample:
        del index
        evicted = torch.randn(self.max_heap_size, self.hidden_size, generator=self.generator)
        heap = torch.randn(self.max_heap_size, self.hidden_size, generator=self.generator)
        teacher_scores = torch.softmax(torch.randn(self.max_heap_size, generator=self.generator), dim=0)
        return HiddenStateSample(
            backbone=self.backbone,
            evicted_vectors=evicted,
            heap_vectors=heap,
            teacher_scores=teacher_scores,
        )
