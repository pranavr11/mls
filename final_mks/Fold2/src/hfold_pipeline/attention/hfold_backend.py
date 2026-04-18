from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, MutableMapping
from typing import Any, Protocol

from ..config import HFoldConfig


class HFoldBackendProtocol(Protocol):
    def __call__(
        self,
        *,
        base_attention: Any,
        original_forward: Callable[..., Any],
        bound_arguments: MutableMapping[str, Any],
        sliding_window_attention_mask: Any,
        layer_index: int,
        hfold_config: HFoldConfig,
    ) -> Any:
        ...


def load_hfold_backend(
    backend_spec: str | None,
    hfold_config: HFoldConfig,
) -> HFoldBackendProtocol | None:
    if backend_spec is None:
        return None

    if ":" not in backend_spec:
        raise ValueError(
            "attention.hfold_backend must use the format 'package.module:callable_name'."
        )

    module_name, object_name = backend_spec.split(":", 1)
    module = importlib.import_module(module_name)
    obj = getattr(module, object_name)

    if inspect.isclass(obj):
        backend = obj(hfold_config)
    elif inspect.isfunction(obj):
        try:
            backend = obj(hfold_config)
        except TypeError:
            backend = obj
    else:
        backend = obj

    if not callable(backend):
        raise TypeError(f"Loaded HFold backend '{backend_spec}' is not callable.")

    return backend

