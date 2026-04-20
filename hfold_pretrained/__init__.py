"""HFOLD attention core and Hugging Face backbone patching (GPT-2, GPT-NeoX / Pythia)."""

from .patch import (
    HFoldGPT2Attention,
    HFoldGPTNeoXAttention,
    assert_pretrained_hfold_compatible_transformers,
    replace_gpt2_attention_with_hfold,
    replace_pythia_attention_with_hfold,
)

__all__ = [
    "HFoldGPT2Attention",
    "HFoldGPTNeoXAttention",
    "assert_pretrained_hfold_compatible_transformers",
    "replace_gpt2_attention_with_hfold",
    "replace_pythia_attention_with_hfold",
]
