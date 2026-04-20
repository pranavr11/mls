from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from typing import Any

from ..config import AttentionConfig


class AttentionStrategy(ABC):
    def __init__(self, attention_config: AttentionConfig):
        self.config = attention_config

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def invoke(
        self,
        *,
        module: Any,
        original_forward,
        bound_arguments: MutableMapping[str, Any],
        layer_index: int,
    ) -> Any:
        ...

    def prepare_module(self, *, module: Any, layer_index: int) -> None:
        del module, layer_index
