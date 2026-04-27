"""
Regression tests guarding the critical HFold algorithm invariants under the
global-heap (model-level) implementation.

- Heap vectors are prepended so original tokens see them under causal masking.
- Auxiliary models receive adapter-encoded inputs at inference.
"""
from __future__ import annotations

import torch

from hfold.config.schema import HFoldConfig, HFoldModelConfig
from hfold.inference.hfold_runtime import HFoldRuntime
from hfold.inference.vector_store import append_heap_vectors, split_appended_outputs
from hfold.models.adapters import BackboneAdapterRegistry
from hfold.models.embedding_autoencoder import EmbeddingAutoencoder
from hfold.models.relevancy_transformer import RelevancyTransformer


def test_prepended_heap_visible_to_original_tokens():
    torch.manual_seed(0)
    hidden_size = 4
    seq_len = 3
    heap_len = 2

    base = torch.randn(1, seq_len, hidden_size)
    heap = torch.randn(1, heap_len, hidden_size)

    augmented = append_heap_vectors(base, heap)
    assert augmented.size(1) == seq_len + heap_len
    s = augmented.size(1)
    causal = torch.tril(torch.ones(s, s, dtype=torch.bool))
    for query_offset in range(seq_len):
        assert causal[heap_len + query_offset, :heap_len].tolist().count(True) == heap_len, (
            "all original-token queries must see all prepended heap keys"
        )

    fake_outputs = augmented
    token_out, heap_out = split_appended_outputs(fake_outputs, seq_len)
    assert token_out.shape == (1, seq_len, hidden_size)
    assert heap_out.shape == (1, heap_len, hidden_size)
    assert torch.equal(heap_out, heap)
    assert torch.equal(token_out, base)


def test_runtime_uses_adapters_for_aux_models():
    torch.manual_seed(0)
    hidden_size = 8
    adapter_dim = 16
    config = HFoldConfig(
        model=HFoldModelConfig(
            hidden_size=hidden_size,
            num_heads=2,
            max_heap_size=2,
            top_w=2,
            pop_k=1,
            adapter_dim=adapter_dim,
        )
    )
    runtime = HFoldRuntime(config)
    adapters = BackboneAdapterRegistry(specs={"pythia": hidden_size}, shared_dim=adapter_dim)
    runtime.attach_adapters(adapters, "pythia")
    embed = EmbeddingAutoencoder(hidden_size=adapter_dim, latent_size=adapter_dim, max_slots=config.model.max_heap_size)
    rel = RelevancyTransformer(hidden_size=adapter_dim, num_layers=1, num_heads=2)

    runtime.prime_timestep_zero(
        layer_index=0,
        vectors=torch.randn(2, hidden_size),
        scores=torch.tensor([0.9, 0.6]),
        token_positions=torch.tensor([0, 1]),
        head_indices=torch.tensor([0, 0]),
        time_index=0,
    )
    popped = runtime.pop_top_k(layer_index=0)
    artifacts = runtime.step_with_reinsert_and_fold(
        layer_index=0,
        popped_entries=popped,
        transformed_popped_vectors=torch.randn(len(popped), hidden_size),
        new_vectors=torch.randn(2, hidden_size),
        new_scores=torch.tensor([0.7, 0.5]),
        new_token_positions=torch.tensor([2, 3]),
        new_head_indices=torch.tensor([0, 0]),
        time_index=1,
        embedding_model=embed,
        relevancy_model=rel,
    )
    assert artifacts.summary_embedding is not None
    assert artifacts.summary_embedding.shape[-1] == adapter_dim
    heap = runtime.export_heap_entries(layer_index=0)
    assert heap, "heap should not be empty"
    assert heap[0].vector.shape[-1] == hidden_size
