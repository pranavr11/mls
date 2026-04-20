"""
HFOLD Transformer: Complete decoder-only transformer with HFOLD attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

try:
    from hfold.core.hfold_attention_v2 import (
        HFoldMultiHeadAttention,
        HeapHeadBucket,
        as_heap_bucket,
        copy_heap_bucket_deep,
    )
    from hfold.core.config import HFoldConfig, RMSNorm, FeedForward
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.hfold_attention_v2 import (
        HFoldMultiHeadAttention,
        HeapHeadBucket,
        as_heap_bucket,
        copy_heap_bucket_deep,
    )
    from core.config import HFoldConfig, RMSNorm, FeedForward


class HFoldTransformerLayer(nn.Module):
    """Single transformer layer with HFOLD attention"""

    def __init__(self, config: HFoldConfig):
        super().__init__()

        self.self_attn = HFoldMultiHeadAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            window_size=config.window_size,
            heap_size=config.heap_size,
            q_topk=config.q_topk,
            e_pop=config.e_pop,
            dropout=config.dropout,
        )

        self.ff = FeedForward(
            d_model=config.d_model,
            d_ff=config.d_ff,
            activation=config.activation,
            dropout=config.dropout,
        )

        self.norm1 = RMSNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm2 = RMSNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        token_idx: int,
        heap: Optional[List] = None,
    ):
        attn_output, heap, attn_info = self.self_attn(x, current_token_idx=token_idx, heaps=heap)
        x_current = x[:, -1:, :] + self.dropout(attn_output)
        x_current = self.norm1(x_current)

        ff_output = self.ff(x_current)
        x_current = x_current + self.dropout(ff_output)
        x_current = self.norm2(x_current)

        return x_current, heap, attn_info


class HFoldTransformer(nn.Module):
    """
    Decoder-only LM with HFOLD attention.

    HFold attention per position uses O(window + e_pop) candidates per head (linear in sequence
    length for the attention kernel). Building the full sequence for training uses a prefix
    buffer of shape (batch, seq_len, d_model) per layer; autograd-safe updates clone O(seq_len)
    per token per layer in the current implementation (quadratic total time in seq_len), not the
    ideal O(n) wall-clock forward—only the HFold *kernel* matches the proposal's constant
    candidate count per step.
    """

    def __init__(self, config: HFoldConfig):
        super().__init__()
        self.config = config

        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)

        self.register_buffer("position_ids", torch.arange(config.max_seq_len).unsqueeze(0))
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)

        self.layers = nn.ModuleList([HFoldTransformerLayer(config) for _ in range(config.n_layers)])

        self.final_norm = RMSNorm(config.d_model, eps=config.layer_norm_eps)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight

        self.dropout = nn.Dropout(config.dropout)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def _init_heaps(self, batch_size: int) -> List[List[List[HeapHeadBucket]]]:
        return [
            [[HeapHeadBucket() for _ in range(self.config.n_heads)] for _ in range(batch_size)]
            for _ in range(self.config.n_layers)
        ]

    def _clone_heaps(self, heaps: List) -> List:
        # heaps[layer][batch][head] -> HeapHeadBucket
        return [
            [[copy_heap_bucket_deep(as_heap_bucket(bucket)) for bucket in batch_row] for batch_row in layer_heaps]
            for layer_heaps in heaps
        ]

    def forward(
        self,
        input_ids: torch.Tensor,
        heaps: Optional[List] = None,
        return_logits: bool = True,
        return_heaps: bool = True,
    ) -> Dict:
        """
        When ``heaps`` is not None, it must be state from a compatible prefix of the same
        ``input_ids`` (e.g. continued generation). For a full pass from the start of the
        sequence, use ``heaps=None`` so token indices in the heap match the key cache.
        """
        batch_size, seq_len = input_ids.shape

        if heaps is None:
            new_heaps = self._init_heaps(batch_size)
        else:
            new_heaps = self._clone_heaps(heaps)

        debug_info: Dict = {"layer_heap_sizes": [], "effective_context": self.config.effective_context}

        x = self.token_embed(input_ids)
        pos_ids = self.position_ids[:, :seq_len]
        x = x + self.pos_embed(pos_ids)
        x = self.dropout(x)

        d_model = self.config.d_model
        layer_hidden = [x.new_zeros(batch_size, seq_len, d_model) for _ in range(self.config.n_layers)]

        for token_idx in range(seq_len):
            # Clone prefixes so autograd never sees inplace writes to shared layer buffers
            # as conflicting with strided views from earlier subgraphs.
            layer_input = x[:, : token_idx + 1, :].clone()

            for layer_idx, layer in enumerate(self.layers):
                x_out, new_heaps[layer_idx], _ = layer(
                    layer_input,
                    token_idx=token_idx,
                    heap=new_heaps[layer_idx],
                )
                row = x_out.squeeze(1)
                lh = layer_hidden[layer_idx]
                new_lh = lh.clone()
                new_lh[:, token_idx, :] = row
                layer_hidden[layer_idx] = new_lh

                if layer_idx < len(self.layers) - 1:
                    layer_input = new_lh[:, : token_idx + 1].clone()

        x_out = layer_hidden[-1]
        x_out = self.final_norm(x_out)

        # new_heaps: [layer][batch][head] -> HeapHeadBucket
        debug_info["layer_heap_sizes"] = [
            [[len(bucket) for bucket in batch_row] for batch_row in layer] for layer in new_heaps
        ]

        return {
            "logits": self.lm_head(x_out) if return_logits else None,
            "hidden_states": x_out,
            "heaps": new_heaps if return_heaps else None,
            "debug_info": debug_info,
        }

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 0.95,
        num_return_sequences: int = 1,
    ) -> torch.Tensor:
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            outputs = self.forward(
                generated,
                heaps=None,
                return_logits=True,
                return_heaps=False,
            )

            logits = outputs["logits"]
            next_token_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, float("-inf"))

            if top_p < 1.0:
                for bi in range(next_token_logits.shape[0]):
                    row = next_token_logits[bi : bi + 1]
                    sorted_logits, sorted_indices = torch.sort(row, descending=True, dim=-1)
                    cumsum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    mask = cumsum > top_p
                    mask[..., 0] = False
                    remove_idx = sorted_indices[mask]
                    next_token_logits[bi, remove_idx] = float("-inf")

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            if generated.shape[1] >= self.config.max_seq_len:
                break

        return generated

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
