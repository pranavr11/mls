from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .attention import HFoldAttention
from .config import HFoldConfig, HFoldMemoryState


class HFoldTransformerBlock(nn.Module):
    def __init__(self, config: HFoldConfig) -> None:
        super().__init__()
        hidden_dim = int(config.d_model * config.mlp_ratio)
        self.ln_1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.attn = HFoldAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, config.d_model),
            nn.Dropout(config.dropout_p),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache: Optional[HFoldMemoryState] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[HFoldMemoryState]]:
        attn_input = self.ln_1(hidden_states)
        attn_output, updated_cache = self.attn(
            attn_input,
            attention_mask=attention_mask,
            cache=cache,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))
        return hidden_states, updated_cache
