from __future__ import annotations

import torch


def append_heap_vectors(hidden_states: torch.Tensor, heap_vectors: torch.Tensor) -> torch.Tensor:
    """
    Prepend heap vectors so that under causal masking the original tokens
    can attend to them as additional context.

    hidden_states: [batch, seq, hidden]
    heap_vectors: [batch, k, hidden]
    """
    if heap_vectors.numel() == 0:
        return hidden_states
    return torch.cat([heap_vectors, hidden_states], dim=1)


def split_appended_outputs(outputs: torch.Tensor, original_seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Inverse of `append_heap_vectors` (which prepends):
    - heap outputs are at the front
    - original token outputs are at the back

    Returns:
    - original token outputs [batch, seq, hidden]
    - transformed heap outputs [batch, k, hidden]
    """
    total = outputs.size(1)
    heap_len = total - original_seq_len
    if heap_len <= 0:
        return outputs[:, :original_seq_len, :], outputs.new_zeros((outputs.size(0), 0, outputs.size(-1)))
    transformed_heap = outputs[:, :heap_len, :]
    token_output = outputs[:, heap_len:, :]
    return token_output, transformed_heap
