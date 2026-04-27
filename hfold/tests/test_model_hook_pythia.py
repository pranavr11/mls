"""Tests for the model-level (global-heap) HFold wrapper on a Pythia-shaped trunk."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from hfold.config.schema import HFoldConfig, HFoldModelConfig
from hfold.inference.hfold_runtime import HFoldRuntime
from hfold.inference.model_hook import GLOBAL_HEAP_INDEX, wrap_pythia_with_hfold
from hfold.models.adapters import BackboneAdapterRegistry
from hfold.models.embedding_autoencoder import EmbeddingAutoencoder
from hfold.models.relevancy_transformer import RelevancyTransformer


@dataclass
class _FakeTrunkOutput:
    last_hidden_state: torch.Tensor
    attentions: tuple[torch.Tensor, ...]


class _MultiLayerTrunk(nn.Module):
    """Mimics gpt_neox: has embed_in and processes inputs_embeds through L layers."""

    def __init__(self, hidden_size: int, num_layers: int, vocab_size: int = 32) -> None:
        super().__init__()
        self.embed_in = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])

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
            h = layer(h)
            if output_attentions:
                b, s, _ = h.shape
                attentions.append(torch.softmax(torch.randn(b, 1, s, s, device=h.device), dim=-1))
        return _FakeTrunkOutput(last_hidden_state=h, attentions=tuple(attentions))


class _Pythia(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.gpt_neox = _MultiLayerTrunk(hidden_size, num_layers)


def _make_runtime(hidden_size: int, adapter_dim: int = None) -> HFoldRuntime:
    adapter_dim = adapter_dim or hidden_size
    config = HFoldConfig(
        model=HFoldModelConfig(
            hidden_size=hidden_size,
            num_heads=2,
            max_heap_size=4,
            top_w=2,
            pop_k=2,
            adapter_dim=adapter_dim,
        )
    )
    runtime = HFoldRuntime(config)
    runtime.attach_adapters(BackboneAdapterRegistry(specs={"pythia": hidden_size}, shared_dim=adapter_dim), "pythia")
    return runtime


def test_global_hook_uses_single_heap_regardless_of_layer_count():
    torch.manual_seed(0)
    hidden_size = 8
    model = _Pythia(hidden_size=hidden_size, num_layers=4)
    runtime = _make_runtime(hidden_size)
    embed = EmbeddingAutoencoder(hidden_size=hidden_size, latent_size=hidden_size, max_slots=runtime.config.model.max_heap_size)
    rel = RelevancyTransformer(hidden_size=hidden_size, num_layers=1, num_heads=2)
    wrap_pythia_with_hfold(model, runtime, embed, rel)

    input_ids = torch.randint(0, 16, (1, 6))
    _ = model.gpt_neox(input_ids=input_ids)
    _ = model.gpt_neox(input_ids=input_ids)

    assert list(runtime.state.layers.keys()) == [GLOBAL_HEAP_INDEX]
    assert runtime.state.layers[GLOBAL_HEAP_INDEX].call_count == 2


def test_global_hook_preserves_token_output_shape():
    torch.manual_seed(0)
    hidden_size = 8
    model = _Pythia(hidden_size=hidden_size, num_layers=2)
    runtime = _make_runtime(hidden_size)
    embed = EmbeddingAutoencoder(hidden_size=hidden_size, latent_size=hidden_size, max_slots=runtime.config.model.max_heap_size)
    rel = RelevancyTransformer(hidden_size=hidden_size, num_layers=1, num_heads=2)
    wrap_pythia_with_hfold(model, runtime, embed, rel)

    input_ids = torch.randint(0, 16, (1, 5))
    out0 = model.gpt_neox(input_ids=input_ids)
    assert out0.last_hidden_state.shape == (1, 5, hidden_size)
    out1 = model.gpt_neox(input_ids=input_ids)
    # After step 1: heap vectors are prepended internally but the hook slices
    # the heap-rows out of last_hidden_state before returning, so the LM head
    # downstream still sees seq_len-many positions.
    assert out1.last_hidden_state.shape == (1, 5, hidden_size)


def test_global_hook_uses_adapters_for_aux_models():
    torch.manual_seed(0)
    hidden_size = 8
    adapter_dim = 16
    model = _Pythia(hidden_size=hidden_size, num_layers=2)
    runtime = _make_runtime(hidden_size, adapter_dim=adapter_dim)
    embed = EmbeddingAutoencoder(hidden_size=adapter_dim, latent_size=adapter_dim, max_slots=runtime.config.model.max_heap_size)
    rel = RelevancyTransformer(hidden_size=adapter_dim, num_layers=1, num_heads=2)
    wrap_pythia_with_hfold(model, runtime, embed, rel)

    input_ids = torch.randint(0, 16, (1, 6))
    _ = model.gpt_neox(input_ids=input_ids)
    _ = model.gpt_neox(input_ids=input_ids)

    heap = runtime.export_heap_entries(layer_index=GLOBAL_HEAP_INDEX)
    if heap:
        assert heap[0].vector.shape[-1] == hidden_size  # decoded back to backbone dim
