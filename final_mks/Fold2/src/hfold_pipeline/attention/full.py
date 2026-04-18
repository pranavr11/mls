from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from .base import AttentionStrategy


class PythiaFullAttention(AttentionStrategy):
    @property
    def name(self) -> str:
        return "full"

    def invoke(
        self,
        *,
        module: Any,
        original_forward,
        bound_arguments: MutableMapping[str, Any],
        layer_index: int,
    ) -> Any:
        del module, layer_index
        return original_forward(**bound_arguments)

