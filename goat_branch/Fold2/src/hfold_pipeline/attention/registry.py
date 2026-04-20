from __future__ import annotations

from ..config import AttentionConfig, normalize_attention_type
from .full import PythiaFullAttention
from .hfold import PythiaHFoldAttention
from .native_hfold_backend import build_native_hfold_backend
from .sliding_window import PythiaSlidingWindowAttention


def build_attention_strategy(attention_config: AttentionConfig, *, hfold_backend=None):
    attention_type = normalize_attention_type(attention_config.attention_type)
    attention_config.attention_type = attention_type

    if attention_type == "full":
        return PythiaFullAttention(attention_config)
    if attention_type == "sliding_window":
        return PythiaSlidingWindowAttention(attention_config)
    if attention_type == "hfold":
        if hfold_backend is None:
            hfold_backend = build_native_hfold_backend(attention_config)
        return PythiaHFoldAttention(attention_config, hfold_backend)
    raise ValueError(f"Unsupported attention type: {attention_type}")
