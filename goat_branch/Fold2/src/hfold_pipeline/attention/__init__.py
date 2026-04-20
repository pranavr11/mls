"""Attention utilities for full, sliding-window, and native HFold integration."""

from .base import AttentionStrategy
from .full import PythiaFullAttention
from .hfold import PythiaHFoldAttention
from .hfold_backend import HFoldBackendProtocol, load_hfold_backend
from .masking import build_sliding_window_attention_mask, infer_past_key_values_length
from .native_hfold_backend import NativeHFoldBackend, build_native_hfold_backend
from .native_hfold_core import HFoldMemoryState, NativeHFoldCore
from .registry import build_attention_strategy
from .sliding_window import PythiaSlidingWindowAttention

__all__ = [
    "AttentionStrategy",
    "HFoldBackendProtocol",
    "HFoldMemoryState",
    "NativeHFoldBackend",
    "NativeHFoldCore",
    "PythiaFullAttention",
    "PythiaHFoldAttention",
    "PythiaSlidingWindowAttention",
    "build_sliding_window_attention_mask",
    "build_attention_strategy",
    "build_native_hfold_backend",
    "infer_past_key_values_length",
    "load_hfold_backend",
]
