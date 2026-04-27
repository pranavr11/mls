"""Helpers for loading checkpoints saved from wrapped / patched models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def _load_raw_state_dict(checkpoint_dir: Path) -> dict[str, torch.Tensor] | None:
    bin_f = checkpoint_dir / "pytorch_model.bin"
    if bin_f.is_file():
        return torch.load(bin_f, map_location="cpu", weights_only=True)
    saf = checkpoint_dir / "model.safetensors"
    if saf.is_file():
        try:
            from safetensors.torch import load_file

            return load_file(str(saf))
        except ImportError:
            return None
    return None


def remap_gpt_neox_sliding_wrapper_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Map SlidingWindowAttentionWrapper keys to plain GPTNeoXAttention keys."""
    out: dict[str, Any] = {}
    for k, v in state_dict.items():
        nk = k.replace(".attention.original_attention.", ".attention.")
        out[nk] = v
    return out


def load_gpt_neox_causal_lm_from_folder(
    checkpoint_path: str,
    *,
    cache_dir: str,
) -> torch.nn.Module:
    """
    Load a GPT-NeoX causal LM from a local folder.

    If the checkpoint was saved with ``SlidingWindowAttentionWrapper`` (keys like
    ``...attention.original_attention...``), remap so weights bind to a standard
    ``GPTNeoXAttention`` module.
    """
    cp = Path(checkpoint_path)
    config = AutoConfig.from_pretrained(str(cp), cache_dir=cache_dir)
    # Be explicit so behavior is stable across Transformers versions.
    setattr(config, "_attn_implementation", "eager")
    if hasattr(config, "attn_implementation"):
        setattr(config, "attn_implementation", "eager")
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    sd = _load_raw_state_dict(cp)
    if sd is None:
        return AutoModelForCausalLM.from_pretrained(
            str(cp), cache_dir=cache_dir, attn_implementation="eager"
        )
    if any(".attention.original_attention." in k for k in sd):
        sd = remap_gpt_neox_sliding_wrapper_state_dict(sd)
    model.load_state_dict(sd, strict=False)
    return model
