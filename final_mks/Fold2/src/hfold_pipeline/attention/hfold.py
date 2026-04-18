from __future__ import annotations

import warnings
from collections.abc import MutableMapping
from typing import Any

from .base import AttentionStrategy
from .masking import build_sliding_window_attention_mask
from .sliding_window import _get_hidden_states, _get_layer_past


class PythiaHFoldAttention(AttentionStrategy):
    def __init__(self, attention_config, backend):
        super().__init__(attention_config)
        self.backend = backend

    @property
    def name(self) -> str:
        return "hfold"

    def prepare_module(self, *, module: Any, layer_index: int) -> None:
        if self.backend is None:
            return
        prepare = getattr(self.backend, "prepare_module", None)
        if callable(prepare):
            prepare(base_attention=module, layer_index=layer_index)

    def invoke(
        self,
        *,
        module: Any,
        original_forward,
        bound_arguments: MutableMapping[str, Any],
        layer_index: int,
    ) -> Any:
        hidden_states = _get_hidden_states(bound_arguments)
        layer_past = _get_layer_past(bound_arguments)
        original_attention_mask = bound_arguments.get("attention_mask")
        sliding_mask = build_sliding_window_attention_mask(
            hidden_states=hidden_states,
            window_size=self.config.window_size,
            attention_mask=original_attention_mask,
            layer_past=layer_past,
        )
        backend_arguments = dict(bound_arguments)
        backend_arguments["_hfold_token_attention_mask"] = original_attention_mask
        backend_arguments["attention_mask"] = sliding_mask

        if self.backend is None:
            message = (
                "HFold attention was requested, but no backend was provided via "
                "attention.hfold_backend."
            )
            if self.config.allow_hfold_fallback:
                warnings.warn(
                    message + " Falling back to sliding-window attention.",
                    stacklevel=2,
                )
                bound_arguments["attention_mask"] = sliding_mask
                return original_forward(**bound_arguments)
            raise NotImplementedError(message)

        return self.backend(
            base_attention=module,
            original_forward=original_forward,
            bound_arguments=backend_arguments,
            sliding_window_attention_mask=sliding_mask,
            layer_index=layer_index,
            hfold_config=self.config.hfold,
        )
