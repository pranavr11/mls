from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from .base import AttentionStrategy
from .masking import build_sliding_window_attention_mask


class PythiaSlidingWindowAttention(AttentionStrategy):
    @property
    def name(self) -> str:
        return "sliding_window"

    def invoke(
        self,
        *,
        module: Any,
        original_forward,
        bound_arguments: MutableMapping[str, Any],
        layer_index: int,
    ) -> Any:
        del module, layer_index
        hidden_states = _get_hidden_states(bound_arguments)
        layer_past = _get_layer_past(bound_arguments)
        bound_arguments["attention_mask"] = build_sliding_window_attention_mask(
            hidden_states=hidden_states,
            window_size=self.config.window_size,
            attention_mask=bound_arguments.get("attention_mask"),
            layer_past=layer_past,
        )
        return original_forward(**bound_arguments)


def _get_hidden_states(bound_arguments: MutableMapping[str, Any]):
    hidden_states = bound_arguments.get("hidden_states")
    if hidden_states is None:
        raise ValueError("Could not locate hidden_states in attention bound arguments.")
    return hidden_states


def _get_layer_past(bound_arguments: MutableMapping[str, Any]):
    for key in ("layer_past", "past_key_value", "cache", "past_key_values"):
        if key in bound_arguments:
            return bound_arguments[key]
    return None

