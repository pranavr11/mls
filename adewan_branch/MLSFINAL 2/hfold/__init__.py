"""
HFOLD package: use hfold_attention_v2 for the canonical HFold kernel.
"""

from .core.config import HFoldConfig, TrainingConfig
from .core.hfold_attention_v2 import (
    HFoldAttentionV2,
    HFoldMultiHeadAttention,
    HeapHeadBucket,
)
from .models.hfold_transformer import HFoldTransformer, HFoldTransformerLayer

__all__ = [
    "HFoldConfig",
    "TrainingConfig",
    "HFoldAttentionV2",
    "HFoldMultiHeadAttention",
    "HeapHeadBucket",
    "HFoldTransformer",
    "HFoldTransformerLayer",
]
