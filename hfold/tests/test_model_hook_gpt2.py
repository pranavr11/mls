"""Tests for the model-level (global-heap) HFold wrapper on a GPT-2-shaped trunk."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from hfold.config.schema import HFoldConfig, HFoldModelConfig
from hfold.inference.hfold_runtime import HFoldRuntime
from hfold.inference.model_hook import GLOBAL_HEAP_INDEX, wrap_gpt2_with_hfold
from hfold.models.adapters import BackboneAdapterRegistry
from hfold.models.embedding_autoencoder import EmbeddingAutoencoder
from hfold.models.relevancy_transformer import RelevancyTransformer


@dataclass
class _FakeTrunkOutput:
    last_hidden_state: torch.Tensor
    attentions: tuple[torch.Tensor, ...]


class _GPT2Trunk(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, vocab_size: int = 32) -> None:
        super().__init__()
        self.wte = nn.Embedding(vocab_size, hidden_size)
        self.h = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])

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
        h = inputs_embeds if inputs_embeds is not None else self.wte(input_ids)
        attentions = []
        for layer in self.h:
            h = layer(h)
            if output_attentions:
                b, s, _ = h.shape
                attentions.append(torch.softmax(torch.randn(b, 1, s, s, device=h.device), dim=-1))
        return _FakeTrunkOutput(last_hidden_state=h, attentions=tuple(attentions))


class _GPT2(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.transformer = _GPT2Trunk(hidden_size, num_layers)


def test_global_hook_gpt2_single_heap():
    torch.manual_seed(0)
    hidden_size = 8
    model = _GPT2(hidden_size=hidden_size, num_layers=3)
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
    runtime.attach_adapters(BackboneAdapterRegistry(specs={"gpt2": hidden_size}, shared_dim=hidden_size), "gpt2")
    embed = EmbeddingAutoencoder(hidden_size=hidden_size, latent_size=hidden_size, max_slots=config.model.max_heap_size)
    rel = RelevancyTransformer(hidden_size=hidden_size, num_layers=1, num_heads=2)
    wrap_gpt2_with_hfold(model, runtime, embed, rel)

    input_ids = torch.randint(0, 16, (1, 6))
    _ = model.transformer(input_ids=input_ids)
    _ = model.transformer(input_ids=input_ids)

    assert list(runtime.state.layers.keys()) == [GLOBAL_HEAP_INDEX]
    assert runtime.state.layers[GLOBAL_HEAP_INDEX].call_count == 2
