from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class HFoldHeapEntry:
    score: float
    vector: torch.Tensor
    token_position: int
    layer_index: int
    head_index: int
    time_index: int
    source: str = "local"
    id: int = 0


@dataclass
class HFoldLayerState:
    layer_index: int
    heap: list[HFoldHeapEntry] = field(default_factory=list)
    next_entry_id: int = 0
    call_count: int = 0


@dataclass
class HFoldRuntimeState:
    layers: dict[int, HFoldLayerState] = field(default_factory=dict)
    timestep: int = 0
