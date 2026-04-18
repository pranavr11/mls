from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class HFoldConfig:
    d_model: int
    n_heads: int
    window_size: int
    heap_size: int
    top_q: int
    retrieve_e: int
    dropout_p: float = 0.0
    attention_dropout_p: float = 0.0
    layer_norm_eps: float = 1e-5
    mlp_ratio: float = 4.0
    fold_strategy: str = "gated_residual"

    @property
    def head_dim(self) -> int:
        if self.d_model % self.n_heads != 0:
            msg = "d_model must be divisible by n_heads."
            raise ValueError(msg)
        return self.d_model // self.n_heads


@dataclass
class HFoldMemoryEntry:
    state: torch.Tensor
    score: torch.Tensor
    source_position: torch.Tensor
    fold_count: torch.Tensor
    valid: torch.Tensor


@dataclass
class HFoldMemoryState:
    memory_states: torch.Tensor
    memory_scores: torch.Tensor
    memory_positions: torch.Tensor
    memory_fold_counts: torch.Tensor
    memory_valid_mask: torch.Tensor
    local_keys: torch.Tensor
    local_values: torch.Tensor
    local_positions: torch.Tensor
    local_valid_mask: torch.Tensor
    next_positions: torch.Tensor
    last_candidate_counts: Optional[torch.Tensor] = None
    last_retrieved_counts: Optional[torch.Tensor] = None
    last_removed_counts: Optional[torch.Tensor] = None
    last_evicted_counts: Optional[torch.Tensor] = None

    @classmethod
    def empty(
        cls,
        batch_size: int,
        config: HFoldConfig,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "HFoldMemoryState":
        head_dim = config.head_dim
        memory_shape = (batch_size, config.n_heads, config.heap_size, head_dim)
        memory_score_shape = (batch_size, config.n_heads, config.heap_size)
        local_shape = (batch_size, config.n_heads, config.window_size, head_dim)
        local_pos_shape = (batch_size, config.window_size)

        return cls(
            memory_states=torch.zeros(memory_shape, device=device, dtype=dtype),
            memory_scores=torch.zeros(memory_score_shape, device=device, dtype=dtype),
            memory_positions=torch.full(
                memory_score_shape,
                fill_value=-1,
                device=device,
                dtype=torch.long,
            ),
            memory_fold_counts=torch.zeros(
                memory_score_shape,
                device=device,
                dtype=torch.long,
            ),
            memory_valid_mask=torch.zeros(
                memory_score_shape,
                device=device,
                dtype=torch.bool,
            ),
            local_keys=torch.zeros(local_shape, device=device, dtype=dtype),
            local_values=torch.zeros(local_shape, device=device, dtype=dtype),
            local_positions=torch.full(
                local_pos_shape,
                fill_value=-1,
                device=device,
                dtype=torch.long,
            ),
            local_valid_mask=torch.zeros(
                local_pos_shape,
                device=device,
                dtype=torch.bool,
            ),
            next_positions=torch.zeros(batch_size, device=device, dtype=torch.long),
        )

    def to(self, *, device: torch.device, dtype: torch.dtype) -> "HFoldMemoryState":
        return HFoldMemoryState(
            memory_states=self.memory_states.to(device=device, dtype=dtype),
            memory_scores=self.memory_scores.to(device=device, dtype=dtype),
            memory_positions=self.memory_positions.to(device=device),
            memory_fold_counts=self.memory_fold_counts.to(device=device),
            memory_valid_mask=self.memory_valid_mask.to(device=device),
            local_keys=self.local_keys.to(device=device, dtype=dtype),
            local_values=self.local_values.to(device=device, dtype=dtype),
            local_positions=self.local_positions.to(device=device),
            local_valid_mask=self.local_valid_mask.to(device=device),
            next_positions=self.next_positions.to(device=device),
            last_candidate_counts=None
            if self.last_candidate_counts is None
            else self.last_candidate_counts.to(device=device),
            last_retrieved_counts=None
            if self.last_retrieved_counts is None
            else self.last_retrieved_counts.to(device=device),
            last_removed_counts=None
            if self.last_removed_counts is None
            else self.last_removed_counts.to(device=device),
            last_evicted_counts=None
            if self.last_evicted_counts is None
            else self.last_evicted_counts.to(device=device),
        )
