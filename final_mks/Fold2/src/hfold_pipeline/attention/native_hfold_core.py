from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


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
    last_candidate_counts: torch.Tensor | None = None
    last_retrieved_counts: torch.Tensor | None = None
    last_removed_counts: torch.Tensor | None = None
    last_evicted_counts: torch.Tensor | None = None

    @classmethod
    def empty(
        cls,
        batch_size: int,
        *,
        num_heads: int,
        head_dim: int,
        window_size: int,
        heap_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "HFoldMemoryState":
        memory_shape = (batch_size, num_heads, heap_size, head_dim)
        memory_score_shape = (batch_size, num_heads, heap_size)
        local_shape = (batch_size, num_heads, window_size, head_dim)
        local_pos_shape = (batch_size, window_size)

        return cls(
            memory_states=torch.zeros(memory_shape, device=device, dtype=dtype),
            memory_scores=torch.zeros(memory_score_shape, device=device, dtype=dtype),
            memory_positions=torch.full(memory_score_shape, -1, device=device, dtype=torch.long),
            memory_fold_counts=torch.zeros(memory_score_shape, device=device, dtype=torch.long),
            memory_valid_mask=torch.zeros(memory_score_shape, device=device, dtype=torch.bool),
            local_keys=torch.zeros(local_shape, device=device, dtype=dtype),
            local_values=torch.zeros(local_shape, device=device, dtype=dtype),
            local_positions=torch.full(local_pos_shape, -1, device=device, dtype=torch.long),
            local_valid_mask=torch.zeros(local_pos_shape, device=device, dtype=torch.bool),
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


class NativeHFoldCore(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        window_size: int,
        heap_size: int,
        top_q: int,
        retrieve_e: int,
        attention_dropout_p: float = 0.0,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size
        self.heap_size = heap_size
        self.top_q = top_q
        self.retrieve_e = retrieve_e
        self.scale = self.head_dim**-0.5

        self.memory_to_key = nn.Linear(self.head_dim, self.head_dim)
        self.memory_to_value = nn.Linear(self.head_dim, self.head_dim)
        self.fold_gate = nn.Linear(2 * self.head_dim, self.head_dim)
        self.keep_state_proj = nn.Linear(self.head_dim, self.head_dim)
        self.removed_state_proj = nn.Linear(self.head_dim, self.head_dim)
        self.state_norm = nn.LayerNorm(self.head_dim, eps=layer_norm_eps)
        self.attn_dropout = nn.Dropout(attention_dropout_p)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        cache: HFoldMemoryState | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, HFoldMemoryState | None]:
        if query.dim() != 4 or key.shape != query.shape or value.shape != query.shape:
            raise ValueError("query, key, and value must all have shape [batch, heads, seq, head_dim].")

        batch_size, _, seq_len, _ = query.shape
        if attention_mask is None:
            token_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=query.device)
        else:
            if attention_mask.shape != (batch_size, seq_len):
                raise ValueError("attention_mask must have shape [batch, seq].")
            token_mask = attention_mask.to(dtype=torch.bool, device=query.device)

        state = self._init_state(
            batch_size=batch_size,
            device=query.device,
            dtype=query.dtype,
            cache=cache,
        )
        outputs: list[torch.Tensor] = []

        for index in range(seq_len):
            token_valid = token_mask[:, index]
            current_q = query[:, :, index, :]
            current_k = key[:, :, index, :]
            current_v = value[:, :, index, :]
            current_positions = state.next_positions.clone()

            local = self._build_local_context(state=state)
            _, local_attn_weights = self._compute_attention(
                query=current_q,
                keys=local["keys"],
                valid_mask=local["valid"],
                token_valid=token_valid,
            )
            local_candidates = self._select_local_candidates(
                local_states=local["states"],
                local_positions=local["positions"],
                local_valid=local["valid"],
                local_attn_weights=local_attn_weights,
                current_positions=current_positions,
                token_valid=token_valid,
            )

            next_local = self._advance_local_window(
                state=state,
                current_k=current_k,
                current_v=current_v,
                current_positions=current_positions,
                token_valid=token_valid,
            )
            state_after_insert, overflow = self._insert_candidates_into_memory(
                state=state,
                candidate_states=local_candidates["states"],
                candidate_scores=local_candidates["scores"],
                candidate_positions=local_candidates["positions"],
                candidate_valid_mask=local_candidates["valid"],
            )
            popped = self._pop_memory(state=state_after_insert, token_valid=token_valid)

            combined_keys = torch.cat([popped["keys"], local["keys"]], dim=2)
            combined_values = torch.cat([popped["values"], local["values"]], dim=2)
            combined_valid = torch.cat([popped["valid"], local["valid"]], dim=2)
            _, attn_weights = self._compute_attention(
                query=current_q,
                keys=combined_keys,
                valid_mask=combined_valid,
                token_valid=token_valid,
            )
            attn_probs = self.attn_dropout(attn_weights)
            context = torch.einsum("bhn,bhnd->bhd", attn_probs, combined_values)
            context = context * token_valid[:, None, None].to(context.dtype)
            outputs.append(context)

            removed_states = torch.cat(
                [popped["states"], overflow["states"], next_local["aged_out_states"]],
                dim=2,
            )
            removed_scores = torch.cat(
                [popped["scores"], overflow["scores"], next_local["aged_out_scores"]],
                dim=2,
            )
            removed_positions = torch.cat(
                [
                    popped["positions"],
                    overflow["positions"],
                    next_local["aged_out_positions"].unsqueeze(1).expand(-1, self.num_heads, -1),
                ],
                dim=2,
            )
            removed_valid = torch.cat(
                [popped["valid"], overflow["valid"], next_local["aged_out_valid"]],
                dim=2,
            )
            state = self._fold_removed_memory_state(
                state=popped["state"],
                removed_states=removed_states,
                removed_scores=removed_scores,
                removed_positions=removed_positions,
                removed_valid=removed_valid,
            )
            state.last_candidate_counts = local_candidates["valid"].sum(dim=-1)
            state.last_retrieved_counts = popped["counts"]
            state.last_removed_counts = removed_valid.sum(dim=-1)
            state.last_evicted_counts = overflow["counts"]
            state.local_keys = next_local["keys"]
            state.local_values = next_local["values"]
            state.local_positions = next_local["positions"]
            state.local_valid_mask = next_local["valid"][:, 0, :]
            state.next_positions = state.next_positions + token_valid.to(torch.long)

        context_tensor = torch.stack(outputs, dim=2)
        output = self._merge_heads(context_tensor)
        return output, state if use_cache else None

    def _init_state(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        cache: HFoldMemoryState | None,
    ) -> HFoldMemoryState:
        if cache is None:
            return HFoldMemoryState.empty(
                batch_size,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                window_size=self.window_size,
                heap_size=self.heap_size,
                device=device,
                dtype=dtype,
            )
        return cache.to(device=device, dtype=dtype)

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = tensor.shape
        return tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

    def _build_local_context(self, *, state: HFoldMemoryState) -> dict[str, torch.Tensor]:
        return {
            "keys": state.local_keys,
            "values": state.local_values,
            "states": state.local_keys.clone(),
            "positions": state.local_positions,
            "valid": state.local_valid_mask.unsqueeze(1).expand(-1, self.num_heads, -1),
        }

    def _select_local_candidates(
        self,
        *,
        local_states: torch.Tensor,
        local_positions: torch.Tensor,
        local_valid: torch.Tensor,
        local_attn_weights: torch.Tensor,
        current_positions: torch.Tensor,
        token_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        candidate_count = min(self.top_q, local_states.size(2))
        batch_size = local_states.size(0)
        if candidate_count == 0:
            empty_states = local_states.new_zeros(batch_size, self.num_heads, 0, self.head_dim)
            empty_scores = local_attn_weights.new_zeros(batch_size, self.num_heads, 0)
            empty_positions = local_positions.new_zeros(batch_size, self.num_heads, 0)
            empty_valid = local_valid.new_zeros(batch_size, self.num_heads, 0)
            return {
                "states": empty_states,
                "scores": empty_scores,
                "positions": empty_positions,
                "valid": empty_valid,
            }

        candidate_mask = self._candidate_mask(
            combined_positions=local_positions.unsqueeze(1).expand(-1, self.num_heads, -1),
            combined_valid=local_valid,
            current_positions=current_positions,
            token_valid=token_valid,
        )
        candidate_scores, candidate_indices = torch.topk(
            local_attn_weights.masked_fill(~candidate_mask, float("-inf")),
            k=candidate_count,
            dim=-1,
        )
        candidate_valid = torch.isfinite(candidate_scores)
        candidate_positions = torch.gather(
            local_positions.unsqueeze(1).expand(-1, self.num_heads, -1),
            dim=2,
            index=candidate_indices,
        )
        gather_index = candidate_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        candidate_states = torch.gather(local_states, dim=2, index=gather_index)
        return {
            "states": candidate_states,
            "scores": candidate_scores.detach(),
            "positions": candidate_positions,
            "valid": candidate_valid,
        }

    def _advance_local_window(
        self,
        *,
        state: HFoldMemoryState,
        current_k: torch.Tensor,
        current_v: torch.Tensor,
        current_positions: torch.Tensor,
        token_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size = current_k.size(0)
        keys = torch.zeros(
            batch_size,
            self.num_heads,
            self.window_size,
            self.head_dim,
            device=current_k.device,
            dtype=current_k.dtype,
        )
        values = torch.zeros_like(keys)
        positions = torch.full(
            (batch_size, self.window_size),
            fill_value=-1,
            device=current_k.device,
            dtype=torch.long,
        )
        valid = torch.zeros(
            batch_size,
            self.num_heads,
            self.window_size,
            device=current_k.device,
            dtype=torch.bool,
        )
        aged_out_states = torch.zeros_like(keys)
        aged_out_positions = torch.full(
            (batch_size, self.window_size),
            fill_value=-1,
            device=current_k.device,
            dtype=torch.long,
        )
        aged_out_valid = torch.zeros(
            batch_size,
            self.num_heads,
            self.window_size,
            device=current_k.device,
            dtype=torch.bool,
        )
        aged_out_scores = torch.zeros(
            batch_size,
            self.num_heads,
            self.window_size,
            device=current_k.device,
            dtype=current_k.dtype,
        )

        if self.window_size == 0:
            return {
                "keys": keys,
                "values": values,
                "positions": positions,
                "valid": valid,
                "aged_out_states": aged_out_states,
                "aged_out_positions": aged_out_positions,
                "aged_out_valid": aged_out_valid,
                "aged_out_scores": aged_out_scores,
            }

        for batch_index in range(batch_size):
            local_indices = state.local_valid_mask[batch_index].nonzero(as_tuple=False).flatten()
            key_pieces: list[torch.Tensor] = []
            value_pieces: list[torch.Tensor] = []
            pos_pieces: list[torch.Tensor] = []

            if local_indices.numel() > 0:
                key_pieces.append(state.local_keys[batch_index, :, local_indices, :])
                value_pieces.append(state.local_values[batch_index, :, local_indices, :])
                pos_pieces.append(state.local_positions[batch_index, local_indices])

            if token_valid[batch_index]:
                key_pieces.append(current_k[batch_index].unsqueeze(1))
                value_pieces.append(current_v[batch_index].unsqueeze(1))
                pos_pieces.append(current_positions[batch_index].view(1))

            if not key_pieces:
                continue

            packed_keys = torch.cat(key_pieces, dim=1)
            packed_values = torch.cat(value_pieces, dim=1)
            packed_positions = torch.cat(pos_pieces, dim=0)
            if packed_positions.numel() > self.window_size:
                aged_out_count = packed_positions.numel() - self.window_size
                aged_out_states[batch_index, :, :aged_out_count, :] = packed_keys[:, :aged_out_count, :]
                aged_out_positions[batch_index, :aged_out_count] = packed_positions[:aged_out_count]
                aged_out_valid[batch_index, :, :aged_out_count] = True
                packed_keys = packed_keys[:, -self.window_size :, :]
                packed_values = packed_values[:, -self.window_size :, :]
                packed_positions = packed_positions[-self.window_size :]

            count = packed_positions.numel()
            keys[batch_index, :, :count, :] = packed_keys
            values[batch_index, :, :count, :] = packed_values
            positions[batch_index, :count] = packed_positions
            valid[batch_index, :, :count] = True

        return {
            "keys": keys,
            "values": values,
            "positions": positions,
            "valid": valid,
            "aged_out_states": aged_out_states,
            "aged_out_positions": aged_out_positions,
            "aged_out_valid": aged_out_valid,
            "aged_out_scores": aged_out_scores,
        }

    def _insert_candidates_into_memory(
        self,
        *,
        state: HFoldMemoryState,
        candidate_states: torch.Tensor,
        candidate_scores: torch.Tensor,
        candidate_positions: torch.Tensor,
        candidate_valid_mask: torch.Tensor,
    ) -> tuple[HFoldMemoryState, dict[str, torch.Tensor]]:
        batch_size = state.memory_states.size(0)
        overflow_slots = candidate_states.size(2)

        if self.heap_size == 0:
            empty_states = candidate_states.new_zeros(batch_size, self.num_heads, overflow_slots, self.head_dim)
            empty_scores = candidate_scores.new_zeros(batch_size, self.num_heads, overflow_slots)
            empty_positions = candidate_positions.new_full((batch_size, self.num_heads, overflow_slots), -1)
            empty_valid = candidate_valid_mask.new_zeros(batch_size, self.num_heads, overflow_slots)
            return state, {
                "states": empty_states,
                "scores": empty_scores,
                "positions": empty_positions,
                "valid": empty_valid,
                "counts": empty_valid.sum(dim=-1),
            }

        new_memory_states = torch.zeros_like(state.memory_states)
        new_memory_scores = torch.zeros_like(state.memory_scores)
        new_memory_positions = torch.full_like(state.memory_positions, fill_value=-1)
        new_memory_fold_counts = torch.zeros_like(state.memory_fold_counts)
        new_memory_valid_mask = torch.zeros_like(state.memory_valid_mask)

        overflow_states = candidate_states.new_zeros(batch_size, self.num_heads, overflow_slots, self.head_dim)
        overflow_scores = candidate_scores.new_zeros(batch_size, self.num_heads, overflow_slots)
        overflow_positions = candidate_positions.new_full((batch_size, self.num_heads, overflow_slots), -1)
        overflow_valid = candidate_valid_mask.new_zeros(batch_size, self.num_heads, overflow_slots)

        for batch_index in range(batch_size):
            for head_index in range(self.num_heads):
                base_valid = state.memory_valid_mask[batch_index, head_index]
                cand_valid = candidate_valid_mask[batch_index, head_index]

                base_states = state.memory_states[batch_index, head_index, base_valid, :]
                base_scores = state.memory_scores[batch_index, head_index, base_valid]
                base_positions = state.memory_positions[batch_index, head_index, base_valid]
                base_fold_counts = state.memory_fold_counts[batch_index, head_index, base_valid]

                cand_states = candidate_states[batch_index, head_index, cand_valid, :]
                cand_scores = candidate_scores[batch_index, head_index, cand_valid]
                cand_positions = candidate_positions[batch_index, head_index, cand_valid]
                cand_fold_counts = torch.zeros_like(cand_positions)

                pool_states = torch.cat([base_states, cand_states], dim=0)
                pool_scores = torch.cat([base_scores, cand_scores], dim=0)
                pool_positions = torch.cat([base_positions, cand_positions], dim=0)
                pool_fold_counts = torch.cat([base_fold_counts, cand_fold_counts], dim=0)

                if pool_scores.numel() == 0:
                    continue

                retained_count = min(self.heap_size, pool_scores.numel())
                retained_scores, retained_indices = torch.topk(pool_scores, k=retained_count)
                retained_states = pool_states[retained_indices]
                retained_positions = pool_positions[retained_indices]
                retained_fold_counts = pool_fold_counts[retained_indices]

                order = torch.argsort(retained_scores, descending=True)
                retained_states = retained_states[order]
                retained_scores = retained_scores[order]
                retained_positions = retained_positions[order]
                retained_fold_counts = retained_fold_counts[order]

                overflow_mask = torch.ones(pool_scores.size(0), device=pool_scores.device, dtype=torch.bool)
                overflow_mask[retained_indices] = False
                overflow_count = int(overflow_mask.sum().item())
                if overflow_count > 0:
                    raw_overflow_scores = pool_scores[overflow_mask]
                    raw_overflow_states = pool_states[overflow_mask]
                    raw_overflow_positions = pool_positions[overflow_mask]
                    overflow_order = torch.argsort(raw_overflow_scores, descending=True)
                    overflow_states[batch_index, head_index, :overflow_count, :] = raw_overflow_states[overflow_order]
                    overflow_scores[batch_index, head_index, :overflow_count] = raw_overflow_scores[overflow_order]
                    overflow_positions[batch_index, head_index, :overflow_count] = raw_overflow_positions[overflow_order]
                    overflow_valid[batch_index, head_index, :overflow_count] = True

                new_memory_states[batch_index, head_index, :retained_count, :] = retained_states
                new_memory_scores[batch_index, head_index, :retained_count] = retained_scores
                new_memory_positions[batch_index, head_index, :retained_count] = retained_positions
                new_memory_fold_counts[batch_index, head_index, :retained_count] = retained_fold_counts
                new_memory_valid_mask[batch_index, head_index, :retained_count] = True

        updated_state = HFoldMemoryState(
            memory_states=new_memory_states,
            memory_scores=new_memory_scores,
            memory_positions=new_memory_positions,
            memory_fold_counts=new_memory_fold_counts,
            memory_valid_mask=new_memory_valid_mask,
            local_keys=state.local_keys,
            local_values=state.local_values,
            local_positions=state.local_positions,
            local_valid_mask=state.local_valid_mask,
            next_positions=state.next_positions,
            last_candidate_counts=state.last_candidate_counts,
            last_retrieved_counts=state.last_retrieved_counts,
            last_removed_counts=state.last_removed_counts,
            last_evicted_counts=state.last_evicted_counts,
        )
        return updated_state, {
            "states": overflow_states,
            "scores": overflow_scores,
            "positions": overflow_positions,
            "valid": overflow_valid,
            "counts": overflow_valid.sum(dim=-1),
        }

    def _fold_removed_memory_state(
        self,
        *,
        state: HFoldMemoryState,
        removed_states: torch.Tensor,
        removed_scores: torch.Tensor,
        removed_positions: torch.Tensor,
        removed_valid: torch.Tensor,
    ) -> HFoldMemoryState:
        (
            memory_states,
            memory_scores,
            memory_positions,
            memory_fold_counts,
            memory_valid_mask,
            _,
        ) = self._fold_removed_back_into_memory(
            memory_states=state.memory_states,
            memory_scores=state.memory_scores,
            memory_positions=state.memory_positions,
            memory_fold_counts=state.memory_fold_counts,
            memory_valid_mask=state.memory_valid_mask,
            removed_states=removed_states,
            removed_scores=removed_scores,
            removed_positions=removed_positions,
            removed_valid_mask=removed_valid,
        )
        return HFoldMemoryState(
            memory_states=memory_states,
            memory_scores=memory_scores,
            memory_positions=memory_positions,
            memory_fold_counts=memory_fold_counts,
            memory_valid_mask=memory_valid_mask,
            local_keys=state.local_keys,
            local_values=state.local_values,
            local_positions=state.local_positions,
            local_valid_mask=state.local_valid_mask,
            next_positions=state.next_positions,
            last_candidate_counts=state.last_candidate_counts,
            last_retrieved_counts=state.last_retrieved_counts,
            last_removed_counts=state.last_removed_counts,
            last_evicted_counts=state.last_evicted_counts,
        )

    def _pop_memory(
        self,
        *,
        state: HFoldMemoryState,
        token_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor | HFoldMemoryState]:
        batch_size = state.memory_states.size(0)
        pop_count = min(self.retrieve_e, self.heap_size)
        empty_counts = state.memory_positions.new_zeros(batch_size, self.num_heads)

        if pop_count == 0:
            empty_states = state.memory_states.new_zeros(batch_size, self.num_heads, 0, self.head_dim)
            empty_valid = state.memory_valid_mask.new_zeros(batch_size, self.num_heads, 0)
            return {
                "state": state,
                "states": empty_states,
                "keys": empty_states.clone(),
                "values": empty_states.clone(),
                "scores": state.memory_scores.new_zeros(batch_size, self.num_heads, 0),
                "positions": state.memory_positions.new_zeros(batch_size, self.num_heads, 0),
                "valid": empty_valid,
                "counts": empty_counts,
            }

        popped_states = state.memory_states.new_zeros(batch_size, self.num_heads, pop_count, self.head_dim)
        popped_scores = state.memory_scores.new_zeros(batch_size, self.num_heads, pop_count)
        popped_positions = state.memory_positions.new_full((batch_size, self.num_heads, pop_count), -1)
        popped_valid = state.memory_valid_mask.new_zeros(batch_size, self.num_heads, pop_count)

        new_memory_states = torch.zeros_like(state.memory_states)
        new_memory_scores = torch.zeros_like(state.memory_scores)
        new_memory_positions = torch.full_like(state.memory_positions, fill_value=-1)
        new_memory_fold_counts = torch.zeros_like(state.memory_fold_counts)
        new_memory_valid_mask = torch.zeros_like(state.memory_valid_mask)

        for batch_index in range(batch_size):
            if not token_valid[batch_index]:
                new_memory_states[batch_index] = state.memory_states[batch_index]
                new_memory_scores[batch_index] = state.memory_scores[batch_index]
                new_memory_positions[batch_index] = state.memory_positions[batch_index]
                new_memory_fold_counts[batch_index] = state.memory_fold_counts[batch_index]
                new_memory_valid_mask[batch_index] = state.memory_valid_mask[batch_index]
                continue

            for head_index in range(self.num_heads):
                valid_mask = state.memory_valid_mask[batch_index, head_index]
                valid_indices = valid_mask.nonzero(as_tuple=False).flatten()
                valid_count = valid_indices.numel()
                if valid_count == 0:
                    continue

                candidate_count = min(pop_count, valid_count)
                valid_scores = state.memory_scores[batch_index, head_index, valid_indices]
                _, order = torch.topk(valid_scores, k=valid_count)
                sorted_indices = valid_indices[order]
                pop_indices = sorted_indices[:candidate_count]
                keep_indices = sorted_indices[candidate_count:]

                popped_states[batch_index, head_index, :candidate_count, :] = state.memory_states[
                    batch_index,
                    head_index,
                    pop_indices,
                    :,
                ]
                popped_scores[batch_index, head_index, :candidate_count] = state.memory_scores[
                    batch_index,
                    head_index,
                    pop_indices,
                ]
                popped_positions[batch_index, head_index, :candidate_count] = state.memory_positions[
                    batch_index,
                    head_index,
                    pop_indices,
                ]
                popped_valid[batch_index, head_index, :candidate_count] = True

                if keep_indices.numel() == 0:
                    continue

                keep_count = keep_indices.numel()
                new_memory_states[batch_index, head_index, :keep_count, :] = state.memory_states[
                    batch_index,
                    head_index,
                    keep_indices,
                    :,
                ]
                new_memory_scores[batch_index, head_index, :keep_count] = state.memory_scores[
                    batch_index,
                    head_index,
                    keep_indices,
                ]
                new_memory_positions[batch_index, head_index, :keep_count] = state.memory_positions[
                    batch_index,
                    head_index,
                    keep_indices,
                ]
                new_memory_fold_counts[batch_index, head_index, :keep_count] = state.memory_fold_counts[
                    batch_index,
                    head_index,
                    keep_indices,
                ]
                new_memory_valid_mask[batch_index, head_index, :keep_count] = True

        updated_state = HFoldMemoryState(
            memory_states=new_memory_states,
            memory_scores=new_memory_scores,
            memory_positions=new_memory_positions,
            memory_fold_counts=new_memory_fold_counts,
            memory_valid_mask=new_memory_valid_mask,
            local_keys=state.local_keys,
            local_values=state.local_values,
            local_positions=state.local_positions,
            local_valid_mask=state.local_valid_mask,
            next_positions=state.next_positions,
            last_candidate_counts=state.last_candidate_counts,
            last_retrieved_counts=state.last_retrieved_counts,
            last_removed_counts=state.last_removed_counts,
            last_evicted_counts=state.last_evicted_counts,
        )
        projected_keys = self.memory_to_key(popped_states)
        projected_values = self.memory_to_value(popped_states)
        return {
            "state": updated_state,
            "states": popped_states,
            "keys": projected_keys,
            "values": projected_values,
            "scores": popped_scores,
            "positions": popped_positions,
            "valid": popped_valid,
            "counts": popped_valid.sum(dim=-1),
        }

    def _compute_attention(
        self,
        *,
        query: torch.Tensor,
        keys: torch.Tensor,
        valid_mask: torch.Tensor,
        token_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.einsum("bhd,bhnd->bhn", query, keys) * self.scale
        mask = valid_mask & token_valid[:, None, None]
        masked_logits = logits.masked_fill(~mask, -1.0e9)
        attn = torch.softmax(masked_logits, dim=-1)
        attn = attn * mask.to(attn.dtype)
        denom = attn.sum(dim=-1, keepdim=True).clamp_min(1.0e-9)
        return masked_logits, attn / denom

    def _candidate_mask(
        self,
        *,
        combined_positions: torch.Tensor,
        combined_valid: torch.Tensor,
        current_positions: torch.Tensor,
        token_valid: torch.Tensor,
    ) -> torch.Tensor:
        past_only = combined_positions < current_positions[:, None, None]
        valid = combined_valid & past_only
        return valid & token_valid[:, None, None]

    def _fold_removed_back_into_memory(
        self,
        *,
        memory_states: torch.Tensor,
        memory_scores: torch.Tensor,
        memory_positions: torch.Tensor,
        memory_fold_counts: torch.Tensor,
        memory_valid_mask: torch.Tensor,
        removed_states: torch.Tensor,
        removed_scores: torch.Tensor,
        removed_positions: torch.Tensor,
        removed_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        new_memory_states = torch.zeros_like(memory_states)
        new_memory_scores = torch.zeros_like(memory_scores)
        new_memory_positions = torch.full_like(memory_positions, fill_value=-1)
        new_memory_fold_counts = torch.zeros_like(memory_fold_counts)
        new_memory_valid_mask = torch.zeros_like(memory_valid_mask)
        removed_counts = torch.zeros(
            memory_states.size(0),
            self.num_heads,
            device=memory_states.device,
            dtype=torch.long,
        )

        for batch_index in range(memory_states.size(0)):
            for head_index in range(self.num_heads):
                keep_valid = memory_valid_mask[batch_index, head_index]
                retained_states = memory_states[batch_index, head_index, keep_valid, :]
                retained_scores = memory_scores[batch_index, head_index, keep_valid]
                retained_positions = memory_positions[batch_index, head_index, keep_valid]
                retained_fold_counts = memory_fold_counts[batch_index, head_index, keep_valid]
                removed_valid = removed_valid_mask[batch_index, head_index]
                removed_count = removed_valid.sum()
                removed_counts[batch_index, head_index] = removed_count

                if retained_scores.numel() == 0:
                    continue

                if removed_count > 0:
                    retained_states, retained_positions, retained_fold_counts = self._fold_removed_into_retained(
                        retained_states=retained_states,
                        retained_scores=retained_scores,
                        retained_positions=retained_positions,
                        retained_fold_counts=retained_fold_counts,
                        removed_states=removed_states[batch_index, head_index, removed_valid, :],
                        removed_scores=removed_scores[batch_index, head_index, removed_valid],
                        removed_positions=removed_positions[batch_index, head_index, removed_valid],
                        removed_fold_counts=torch.zeros_like(removed_positions[batch_index, head_index, removed_valid]),
                    )

                keep_count = retained_scores.numel()
                new_memory_states[batch_index, head_index, :keep_count, :] = retained_states
                new_memory_scores[batch_index, head_index, :keep_count] = retained_scores
                new_memory_positions[batch_index, head_index, :keep_count] = retained_positions
                new_memory_fold_counts[batch_index, head_index, :keep_count] = retained_fold_counts
                new_memory_valid_mask[batch_index, head_index, :keep_count] = True

        return (
            new_memory_states,
            new_memory_scores,
            new_memory_positions,
            new_memory_fold_counts,
            new_memory_valid_mask,
            removed_counts,
        )

    def _fold_removed_into_retained(
        self,
        *,
        retained_states: torch.Tensor,
        retained_scores: torch.Tensor,
        retained_positions: torch.Tensor,
        retained_fold_counts: torch.Tensor,
        removed_states: torch.Tensor,
        removed_scores: torch.Tensor,
        removed_positions: torch.Tensor,
        removed_fold_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = torch.softmax(removed_scores, dim=0)
        removed_summary = torch.sum(removed_states * weights.unsqueeze(-1), dim=0)
        gate_input = torch.cat(
            [retained_states, removed_summary.unsqueeze(0).expand_as(retained_states)],
            dim=-1,
        )
        gate = torch.sigmoid(self.fold_gate(gate_input))
        mixed_states = gate * self.keep_state_proj(retained_states) + (1.0 - gate) * self.removed_state_proj(
            removed_summary.unsqueeze(0).expand_as(retained_states)
        )
        updated_states = self.state_norm(mixed_states)
        updated_positions = torch.minimum(retained_positions, removed_positions.min())
        removed_total_folds = removed_fold_counts.sum() + removed_scores.numel()
        updated_fold_counts = retained_fold_counts + removed_total_folds
        return updated_states, updated_positions, updated_fold_counts
