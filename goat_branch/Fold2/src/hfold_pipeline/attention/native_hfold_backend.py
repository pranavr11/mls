from __future__ import annotations

import warnings
from collections.abc import MutableMapping
from typing import Any

import torch

from ..config import AttentionConfig
from .native_hfold_core import HFoldMemoryState, NativeHFoldCore


def build_native_hfold_backend(attention_config: AttentionConfig) -> "NativeHFoldBackend":
    return NativeHFoldBackend(attention_config)


class NativeHFoldBackend:
    def __init__(self, attention_config: AttentionConfig):
        self.attention_config = attention_config

    def prepare_module(self, *, base_attention: torch.nn.Module, layer_index: int) -> None:
        del layer_index
        self._ensure_core(base_attention)

    def __call__(
        self,
        *,
        base_attention: Any,
        original_forward,
        bound_arguments: MutableMapping[str, Any],
        sliding_window_attention_mask: Any,
        layer_index: int,
        hfold_config,
    ) -> Any:
        del original_forward, sliding_window_attention_mask, layer_index, hfold_config

        hidden_states = bound_arguments.get("hidden_states")
        if hidden_states is None:
            raise ValueError("HFold backend could not locate hidden_states in the attention arguments.")

        core = self._ensure_core(base_attention)
        token_attention_mask = _resolve_token_attention_mask(
            bound_arguments.get("_hfold_token_attention_mask"),
            hidden_states=hidden_states,
        )
        query, key, value = self._project_qkv(base_attention, hidden_states)
        query, key = _maybe_apply_rotary(
            base_attention,
            query=query,
            key=key,
            value=value,
            bound_arguments=bound_arguments,
        )

        cache = _extract_cache(bound_arguments)
        if cache is not None and not isinstance(cache, HFoldMemoryState):
            warnings.warn(
                "HFold received a non-HFold cache object. Ignoring it and recomputing from the current sequence.",
                stacklevel=2,
            )
            cache = None

        use_cache = bool(bound_arguments.get("use_cache", False))
        context, present = core(
            query,
            key,
            value,
            attention_mask=token_attention_mask,
            cache=cache,
            use_cache=use_cache,
        )
        attn_output = base_attention.dense(context)
        output_attentions = bool(bound_arguments.get("output_attentions", False))
        if output_attentions:
            return attn_output, present, None
        return attn_output, present

    def _ensure_core(self, base_attention: torch.nn.Module) -> NativeHFoldCore:
        existing = getattr(base_attention, "hfold_native_core", None)
        if existing is not None:
            return existing

        hidden_size = _infer_hidden_size(base_attention)
        num_heads = _infer_num_heads(base_attention)
        attention_dropout_p = _infer_attention_dropout_p(base_attention)
        core = NativeHFoldCore(
            hidden_size=hidden_size,
            num_heads=num_heads,
            window_size=self.attention_config.window_size,
            heap_size=self.attention_config.hfold.heap_size,
            top_q=self.attention_config.hfold.top_q,
            retrieve_e=self.attention_config.hfold.pop_e,
            attention_dropout_p=attention_dropout_p,
        )
        parameter = next(base_attention.parameters(), None)
        if parameter is not None:
            core = core.to(device=parameter.device, dtype=parameter.dtype)
        base_attention.hfold_native_core = core
        return base_attention.hfold_native_core

    def _project_qkv(
        self,
        base_attention: torch.nn.Module,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = base_attention.query_key_value(hidden_states)
        batch_size, seq_len, fused_width = qkv.shape
        hidden_size = fused_width // 3
        num_heads = _infer_num_heads(base_attention)
        head_dim = hidden_size // num_heads
        qkv = qkv.view(batch_size, seq_len, num_heads, 3, head_dim).permute(0, 2, 1, 3, 4)
        query = qkv[:, :, :, 0, :]
        key = qkv[:, :, :, 1, :]
        value = qkv[:, :, :, 2, :]
        return query, key, value


def _infer_hidden_size(base_attention: Any) -> int:
    if hasattr(base_attention, "hidden_size"):
        return int(base_attention.hidden_size)
    if hasattr(base_attention, "query_key_value") and hasattr(base_attention.query_key_value, "in_features"):
        return int(base_attention.query_key_value.in_features)
    if hasattr(base_attention, "dense") and hasattr(base_attention.dense, "out_features"):
        return int(base_attention.dense.out_features)
    raise ValueError("Unable to infer hidden size for the patched attention module.")


def _infer_num_heads(base_attention: Any) -> int:
    for attribute in ("num_attention_heads", "num_heads"):
        if hasattr(base_attention, attribute):
            return int(getattr(base_attention, attribute))
    if hasattr(base_attention, "config") and hasattr(base_attention.config, "num_attention_heads"):
        return int(base_attention.config.num_attention_heads)
    raise ValueError("Unable to infer number of attention heads for the patched attention module.")


def _infer_attention_dropout_p(base_attention: Any) -> float:
    dropout = getattr(base_attention, "attention_dropout", None)
    if dropout is None:
        return 0.0
    if hasattr(dropout, "p"):
        return float(dropout.p)
    try:
        return float(dropout)
    except TypeError:
        return 0.0


def _extract_cache(bound_arguments: MutableMapping[str, Any]) -> Any | None:
    for key in ("layer_past", "past_key_value", "cache", "past_key_values"):
        if key in bound_arguments:
            return bound_arguments[key]
    return None


def _resolve_token_attention_mask(
    attention_mask: torch.Tensor | None,
    *,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    batch_size, seq_len = hidden_states.shape[:2]
    if attention_mask is None:
        return torch.ones(batch_size, seq_len, dtype=torch.bool, device=hidden_states.device)

    if attention_mask.dim() == 2:
        return attention_mask.to(device=hidden_states.device, dtype=torch.bool)

    if attention_mask.dim() == 4:
        attention_mask = attention_mask.to(device=hidden_states.device)
        if attention_mask.dtype == torch.bool:
            return attention_mask[:, 0].any(dim=-1)

        mask_value = torch.finfo(attention_mask.dtype).min
        return attention_mask[:, 0].ne(mask_value).any(dim=-1)

    raise ValueError(
        f"Unsupported attention_mask rank {attention_mask.dim()} for HFold token masking."
    )


def _maybe_apply_rotary(
    base_attention: Any,
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bound_arguments: MutableMapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not hasattr(base_attention, "rotary_emb") and bound_arguments.get("position_embeddings") is None:
        return query, key

    try:
        from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb
    except Exception:
        return query, key

    cos, sin = _resolve_rotary_embeddings(
        base_attention,
        value=value,
        key=key,
        bound_arguments=bound_arguments,
    )
    if cos is None or sin is None:
        return query, key

    rotary_ndims = int(getattr(base_attention, "rotary_ndims", query.size(-1)))
    rotary_ndims = min(rotary_ndims, query.size(-1))
    if rotary_ndims <= 0:
        return query, key

    query_rot, query_pass = query[..., :rotary_ndims], query[..., rotary_ndims:]
    key_rot, key_pass = key[..., :rotary_ndims], key[..., rotary_ndims:]
    position_ids = bound_arguments.get("position_ids")

    try:
        rotated_query, rotated_key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
    except TypeError:
        try:
            rotated_query, rotated_key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin)
        except TypeError:
            warnings.warn(
                "HFold could not apply GPT-NeoX rotary embeddings with the installed transformers version. "
                "Continuing without explicit rotary remapping.",
                stacklevel=2,
            )
            return query, key

    return (
        torch.cat([rotated_query, query_pass], dim=-1),
        torch.cat([rotated_key, key_pass], dim=-1),
    )


def _resolve_rotary_embeddings(
    base_attention: Any,
    *,
    value: torch.Tensor,
    key: torch.Tensor,
    bound_arguments: MutableMapping[str, Any],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    position_embeddings = bound_arguments.get("position_embeddings")
    if isinstance(position_embeddings, (tuple, list)) and len(position_embeddings) >= 2:
        return position_embeddings[0], position_embeddings[1]

    rotary_emb = getattr(base_attention, "rotary_emb", None)
    if rotary_emb is None:
        return None, None

    position_ids = bound_arguments.get("position_ids")
    attempts = [
        lambda: rotary_emb(value, position_ids),
        lambda: rotary_emb(value, seq_len=key.size(2)),
        lambda: rotary_emb(key, position_ids),
        lambda: rotary_emb(key, seq_len=key.size(2)),
    ]
    for attempt in attempts:
        try:
            result = attempt()
        except TypeError:
            continue
        if isinstance(result, (tuple, list)) and len(result) >= 2:
            return result[0], result[1]

    return None, None
