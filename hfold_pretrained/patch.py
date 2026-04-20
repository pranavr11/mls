from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

from .native_hfold_core import NativeHFoldCore


def assert_pretrained_hfold_compatible_transformers() -> None:
    """HFOLD patching overrides ``_attn`` on GPT-2 / GPT-NeoX attention modules (eager path)."""
    if not hasattr(GPT2Attention, "_attn"):
        raise RuntimeError(
            "Pretrained HFOLD patching requires a transformers build where GPT2Attention defines _attn "
            "(e.g. transformers>=4.36,<4.45). Install a compatible version, e.g. "
            "`pip install 'transformers>=4.36,<4.45'`."
        )


def attention_mask_to_token_mask(
    attention_mask: Optional[torch.Tensor],
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert HF-style masks to a per-token boolean mask (True = valid position)."""
    if attention_mask is None:
        return torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    if attention_mask.dim() == 2:
        return attention_mask.to(dtype=torch.bool, device=device)
    if attention_mask.dim() == 4:
        m = attention_mask[:, 0, 0, :seq_len]
        return m > -1e4
    if attention_mask.dim() == 3:
        m = attention_mask[:, 0, :seq_len]
        return m > -1e4
    raise ValueError(f"Unsupported attention_mask shape: {tuple(attention_mask.shape)}")


def _ensure_hfold_core(
    module: nn.Module,
    *,
    hidden_size: int,
    num_heads: int,
    window_size: int,
    heap_size: int,
    top_q: int,
    retrieve_e: int,
    attention_dropout_p: float,
) -> NativeHFoldCore:
    existing = getattr(module, "hfold_native_core", None)
    if existing is not None:
        return existing
    core = NativeHFoldCore(
        hidden_size=hidden_size,
        num_heads=num_heads,
        window_size=window_size,
        heap_size=heap_size,
        top_q=top_q,
        retrieve_e=retrieve_e,
        attention_dropout_p=attention_dropout_p,
    )
    ref = next(module.parameters(), None)
    if ref is not None:
        core = core.to(device=ref.device, dtype=ref.dtype)
    module.hfold_native_core = core
    return module.hfold_native_core


def _merged_to_heads(merged: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    b, s, _ = merged.shape
    return merged.view(b, s, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()


class HFoldGPT2Attention(GPT2Attention):
    """GPT-2 self-attention with HFOLD when query/key/value share the same sequence length."""

    def __init__(
        self,
        config,
        is_cross_attention: bool = False,
        layer_idx: Optional[int] = None,
        *,
        window_size: int,
        heap_size: int,
        top_q: int,
        retrieve_e: int,
    ):
        super().__init__(config, is_cross_attention=is_cross_attention, layer_idx=layer_idx)
        self._hfold_window_size = window_size
        self._hfold_heap_size = heap_size
        self._hfold_top_q = top_q
        self._hfold_retrieve_e = retrieve_e

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        if self.is_cross_attention:
            return GPT2Attention._attn(self, query, key, value, attention_mask, head_mask)
        q_len = query.size(-2)
        k_len = key.size(-2)
        if q_len != k_len:
            return GPT2Attention._attn(self, query, key, value, attention_mask, head_mask)

        core = _ensure_hfold_core(
            self,
            hidden_size=self.embed_dim,
            num_heads=self.num_heads,
            window_size=self._hfold_window_size,
            heap_size=self._hfold_heap_size,
            top_q=self._hfold_top_q,
            retrieve_e=self._hfold_retrieve_e,
            attention_dropout_p=self.attn_dropout.p,
        )
        b = query.size(0)
        device = query.device
        token_mask = attention_mask_to_token_mask(attention_mask, batch_size=b, seq_len=q_len, device=device)
        merged, _ = core(query, key, value, attention_mask=token_mask, use_cache=False)
        attn_output = _merged_to_heads(merged, self.num_heads, self.head_dim)
        return attn_output, None


def replace_gpt2_attention_with_hfold(
    model,
    *,
    window_size: int,
    heap_size: int,
    top_q: int,
    retrieve_e: int,
):
    assert_pretrained_hfold_compatible_transformers()
    for layer_idx, block in enumerate(model.transformer.h):
        old_attn = block.attn
        new_attn = HFoldGPT2Attention(
            model.config,
            is_cross_attention=getattr(old_attn, "is_cross_attention", False),
            layer_idx=getattr(old_attn, "layer_idx", layer_idx),
            window_size=window_size,
            heap_size=heap_size,
            top_q=top_q,
            retrieve_e=retrieve_e,
        )
        new_attn.load_state_dict(old_attn.state_dict(), strict=False)
        ref_param = next(old_attn.parameters(), None)
        if ref_param is not None:
            new_attn.to(device=ref_param.device, dtype=ref_param.dtype)
        block.attn = new_attn
    return model


try:
    from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention as _GPTNeoXAttentionBase
except Exception:  # pragma: no cover - optional dependency path
    _GPTNeoXAttentionBase = None


if _GPTNeoXAttentionBase is not None:

    class HFoldGPTNeoXAttention(_GPTNeoXAttentionBase):
        def __init__(
            self,
            config,
            *,
            window_size: int,
            heap_size: int,
            top_q: int,
            retrieve_e: int,
        ):
            super().__init__(config)
            self._hfold_window_size = window_size
            self._hfold_heap_size = heap_size
            self._hfold_top_q = top_q
            self._hfold_retrieve_e = retrieve_e

        def _attn(self, query, key, value, attention_mask=None, head_mask=None):
            q_len = query.size(-2)
            k_len = key.size(-2)
            if q_len != k_len:
                return _GPTNeoXAttentionBase._attn(self, query, key, value, attention_mask, head_mask)

            core = _ensure_hfold_core(
                self,
                hidden_size=self.hidden_size,
                num_heads=self.num_attention_heads,
                window_size=self._hfold_window_size,
                heap_size=self._hfold_heap_size,
                top_q=self._hfold_top_q,
                retrieve_e=self._hfold_retrieve_e,
                attention_dropout_p=self.attention_dropout.p,
            )
            b = query.size(0)
            device = query.device
            token_mask = attention_mask_to_token_mask(
                attention_mask, batch_size=b, seq_len=q_len, device=device
            )
            merged, _ = core(query, key, value, attention_mask=token_mask, use_cache=False)
            attn_output = _merged_to_heads(merged, self.num_attention_heads, self.head_size)
            return attn_output, None

else:
    HFoldGPTNeoXAttention = None  # type: ignore[misc, assignment]


def replace_pythia_attention_with_hfold(
    model,
    *,
    window_size: int,
    heap_size: int,
    top_q: int,
    retrieve_e: int,
):
    assert_pretrained_hfold_compatible_transformers()
    if HFoldGPTNeoXAttention is None:
        raise RuntimeError("GPTNeoXAttention is not available; cannot patch Pythia.")

    if not hasattr(model, "gpt_neox"):
        raise ValueError("Expected GPT-NeoX architecture (e.g. Pythia).")

    for _layer_idx, block in enumerate(model.gpt_neox.layers):
        old_attn = block.attention
        new_attn = HFoldGPTNeoXAttention(
            model.config,
            window_size=window_size,
            heap_size=heap_size,
            top_q=top_q,
            retrieve_e=retrieve_e,
        )
        new_attn.load_state_dict(old_attn.state_dict(), strict=False)
        ref_param = next(old_attn.parameters(), None)
        if ref_param is not None:
            new_attn.to(device=ref_param.device, dtype=ref_param.dtype)
        block.attention = new_attn

    return model
