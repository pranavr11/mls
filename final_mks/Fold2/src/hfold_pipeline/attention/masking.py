from __future__ import annotations

from typing import Any

import torch


def infer_past_key_values_length(layer_past: Any) -> int:
    if layer_past is None:
        return 0

    if hasattr(layer_past, "next_positions"):
        try:
            next_positions = layer_past.next_positions
            if hasattr(next_positions, "numel") and next_positions.numel() > 0:
                return int(next_positions.max().item())
        except (AttributeError, TypeError, ValueError):
            pass

    if hasattr(layer_past, "get_seq_length") and callable(layer_past.get_seq_length):
        try:
            return int(layer_past.get_seq_length())
        except TypeError:
            pass

    if hasattr(layer_past, "seq_length"):
        try:
            return int(layer_past.seq_length)
        except TypeError:
            pass

    if isinstance(layer_past, (tuple, list)) and layer_past:
        first = layer_past[0]
        if hasattr(first, "shape") and len(first.shape) >= 2:
            return int(first.shape[-2])

    if hasattr(layer_past, "shape") and len(layer_past.shape) >= 2:
        return int(layer_past.shape[-2])

    return 0


def _safe_mask_dtype(dtype: torch.dtype) -> torch.dtype:
    return dtype if torch.is_floating_point(torch.empty((), dtype=dtype)) else torch.float32


def _normalize_attention_mask(
    attention_mask: torch.Tensor | None,
    *,
    batch_size: int,
    query_length: int,
    key_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None

    mask_value = torch.finfo(dtype).min

    if attention_mask.dim() == 2:
        keep_mask = attention_mask.to(device=device)
        blocked = keep_mask.eq(0)[:, None, None, :]
        additive = torch.zeros((batch_size, 1, 1, key_length), dtype=dtype, device=device)
        return additive.masked_fill(blocked, mask_value)

    if attention_mask.dim() == 4:
        attention_mask = attention_mask.to(device=device)
        if attention_mask.dtype == torch.bool:
            additive = torch.zeros_like(attention_mask, dtype=dtype, device=device)
            return additive.masked_fill(~attention_mask, mask_value)
        return attention_mask.to(dtype=dtype, device=device)

    raise ValueError(
        f"Unsupported attention_mask rank {attention_mask.dim()}; expected 2D or 4D mask."
    )


def build_sliding_window_attention_mask(
    hidden_states: torch.Tensor,
    window_size: int,
    attention_mask: torch.Tensor | None = None,
    layer_past: Any | None = None,
) -> torch.Tensor:
    batch_size, query_length = hidden_states.shape[:2]
    past_length = infer_past_key_values_length(layer_past)
    key_length = past_length + query_length
    dtype = _safe_mask_dtype(hidden_states.dtype)
    device = hidden_states.device
    mask_value = torch.finfo(dtype).min

    query_positions = torch.arange(
        past_length,
        past_length + query_length,
        device=device,
    ).unsqueeze(-1)
    key_positions = torch.arange(key_length, device=device).unsqueeze(0)

    allowed = (key_positions <= query_positions) & (key_positions > (query_positions - window_size))
    local_blocked = ~allowed

    base_mask = _normalize_attention_mask(
        attention_mask,
        batch_size=batch_size,
        query_length=query_length,
        key_length=key_length,
        dtype=dtype,
        device=device,
    )

    if base_mask is None:
        local_mask = torch.zeros((1, 1, query_length, key_length), dtype=dtype, device=device)
        return local_mask.masked_fill(local_blocked.unsqueeze(0).unsqueeze(0), mask_value)

    return torch.where(
        local_blocked.unsqueeze(0).unsqueeze(0),
        torch.full((), mask_value, dtype=base_mask.dtype, device=device),
        base_mask,
    )
