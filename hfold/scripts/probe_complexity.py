"""Probe per-step time complexity and heap architecture (debug session 01ddb9).

H10: heap-per-layer implementation vs spec "one heap at model boundaries".
H11: per-step time is O(1) wrt sequence length (with sliding-window inner attention),
     so n decoding steps yield O(n) total.
"""
from __future__ import annotations

import time

import torch
from torch import nn

from dataclasses import dataclass

from hfold._debug_log import debug_log
from hfold.config.schema import HFoldConfig, HFoldModelConfig
from hfold.inference.hfold_runtime import HFoldRuntime
from hfold.inference.model_hook import GLOBAL_HEAP_INDEX, wrap_pythia_with_hfold
from hfold.models.adapters import BackboneAdapterRegistry
from hfold.models.embedding_autoencoder import EmbeddingAutoencoder
from hfold.models.relevancy_transformer import RelevancyTransformer


class SlidingCausalAttention(nn.Module):
    def __init__(self, hidden_size: int, window: int) -> None:
        super().__init__()
        self.query_key_value = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.dense = nn.Linear(hidden_size, hidden_size, bias=False)
        self.window = window
        self.scale = hidden_size ** -0.5

    def forward(self, hidden_states, *args, **kwargs):
        del args, kwargs
        b, s, d = hidden_states.shape
        qkv = self.query_key_value(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        scores = torch.einsum("bsd,btd->bst", q, k) * self.scale
        idx = torch.arange(s, device=hidden_states.device)
        diff = idx.unsqueeze(0) - idx.unsqueeze(1)
        causal_and_sliding = (diff <= 0) & (diff >= -self.window)
        scores = scores.masked_fill(~causal_and_sliding, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = self.dense(torch.einsum("bst,btd->bsd", attn, v))
        return out, None, attn.unsqueeze(1)


@dataclass
class _StackedTrunkOutput:
    last_hidden_state: torch.Tensor
    attentions: tuple[torch.Tensor, ...]


class StackedSlidingTrunk(nn.Module):
    """Pythia-shaped trunk: has embed_in and L sliding-attention layers."""

    def __init__(self, hidden_size: int, num_layers: int, window: int, vocab_size: int = 64) -> None:
        super().__init__()
        self.embed_in = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [SlidingCausalAttention(hidden_size, window) for _ in range(num_layers)]
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        output_attentions=False,
        return_dict=True,
        **_kwargs,
    ):
        del attention_mask, return_dict
        h = inputs_embeds if inputs_embeds is not None else self.embed_in(input_ids)
        attentions = []
        for layer in self.layers:
            h, _, attn = layer(h)
            if output_attentions:
                attentions.append(attn)
        return _StackedTrunkOutput(last_hidden_state=h, attentions=tuple(attentions))


class StackedSliding(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, window: int) -> None:
        super().__init__()
        self.gpt_neox = StackedSlidingTrunk(hidden_size, num_layers, window)


def probe_h11_complexity():
    """Measure forward time vs sequence length n with HFold patched on top of a
    fixed-window attention. We time the AVERAGE per-token cost; under sliding-window
    + fixed-size heap, this should stay roughly flat across n.
    """
    torch.manual_seed(0)
    hidden_size = 32
    window = 64
    num_layers = 2
    config = HFoldConfig(
        model=HFoldModelConfig(
            hidden_size=hidden_size,
            num_heads=4,
            max_heap_size=8,
            top_w=4,
            pop_k=4,
            adapter_dim=hidden_size,
        )
    )
    seq_lengths = [32, 64, 128, 256]
    samples_per_length = 5

    for n in seq_lengths:
        model = StackedSliding(hidden_size, num_layers, window=min(window, n))
        runtime = HFoldRuntime(config)
        runtime.attach_adapters(
            BackboneAdapterRegistry(specs={"pythia": hidden_size}, shared_dim=hidden_size),
            "pythia",
        )
        embed = EmbeddingAutoencoder(hidden_size=hidden_size, latent_size=hidden_size, max_slots=config.model.max_heap_size)
        rel = RelevancyTransformer(hidden_size=hidden_size, num_layers=1, num_heads=2)
        wrap_pythia_with_hfold(model, runtime, embed, rel)

        input_ids = torch.randint(0, 32, (1, n))
        # Warm up (timestep 0).
        with torch.no_grad():
            _ = model.gpt_neox(input_ids=input_ids)
        # Time several timestep>=1 forwards.
        durations = []
        with torch.no_grad():
            for _ in range(samples_per_length):
                start = time.perf_counter()
                _ = model.gpt_neox(input_ids=input_ids)
                durations.append(time.perf_counter() - start)
        avg = sum(durations) / len(durations)
        per_token_us = (avg / n) * 1e6
        debug_log(
            hypothesis_id="H11",
            location="hfold/scripts/probe_complexity.py:probe_h11_complexity",
            message=f"forward time at n={n}",
            data={
                "n": n,
                "avg_forward_seconds": avg,
                "per_token_microseconds": per_token_us,
                "samples": samples_per_length,
                "window": min(window, n),
            },
        )


def probe_h10_per_layer_heap():
    """Capture evidence that heap state is per-layer (deviates from spec)."""
    torch.manual_seed(0)
    hidden_size = 16
    config = HFoldConfig(
        model=HFoldModelConfig(
            hidden_size=hidden_size,
            num_heads=2,
            max_heap_size=4,
            top_w=2,
            pop_k=2,
            adapter_dim=hidden_size,
        )
    )
    runtime = HFoldRuntime(config)
    runtime.attach_adapters(
        BackboneAdapterRegistry(specs={"pythia": hidden_size}, shared_dim=hidden_size),
        "pythia",
    )
    model = StackedSliding(hidden_size, num_layers=3, window=16)
    embed = EmbeddingAutoencoder(hidden_size=hidden_size, latent_size=hidden_size, max_slots=config.model.max_heap_size)
    rel = RelevancyTransformer(hidden_size=hidden_size, num_layers=1, num_heads=2)
    wrap_pythia_with_hfold(model, runtime, embed, rel)

    input_ids = torch.randint(0, 32, (1, 8))
    with torch.no_grad():
        _ = model.gpt_neox(input_ids=input_ids)
        _ = model.gpt_neox(input_ids=input_ids)

    debug_log(
        hypothesis_id="H10",
        location="hfold/scripts/probe_complexity.py:probe_h10_per_layer_heap",
        run_id="post-fix-global",
        message="number of independent layer-heaps after running forwards (post-refactor)",
        data={
            "layer_count_in_runtime": len(runtime.state.layers),
            "layer_keys": sorted(runtime.state.layers.keys()),
            "heap_sizes": {k: len(v.heap) for k, v in runtime.state.layers.items()},
            "spec_compliance": "should be exactly one entry: GLOBAL_HEAP_INDEX=0",
            "global_heap_index": GLOBAL_HEAP_INDEX,
            "model_layer_count": 3,
        },
    )


def main() -> None:
    probe_h10_per_layer_heap()
    probe_h11_complexity()


if __name__ == "__main__":
    main()
