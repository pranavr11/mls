import torch

from hfold.config.schema import HFoldConfig, HFoldModelConfig
from hfold.inference.hfold_runtime import HFoldRuntime
from hfold.models.embedding_autoencoder import EmbeddingAutoencoder
from hfold.models.relevancy_transformer import RelevancyTransformer


def _runtime() -> HFoldRuntime:
    config = HFoldConfig(model=HFoldModelConfig(hidden_size=8, num_heads=2, max_heap_size=2, top_w=2, pop_k=2))
    return HFoldRuntime(config)


def test_timestep_zero_only_inserts():
    runtime = _runtime()
    vectors = torch.randn(3, 8)
    scores = torch.tensor([0.9, 0.3, 0.5])
    positions = torch.tensor([0, 1, 2], dtype=torch.long)
    heads = torch.tensor([0, 1, 0], dtype=torch.long)
    artifacts = runtime.prime_timestep_zero(
        layer_index=0,
        vectors=vectors,
        scores=scores,
        token_positions=positions,
        head_indices=heads,
        time_index=0,
    )
    assert artifacts.popped_entries == []
    assert len(runtime.export_heap_entries(layer_index=0)) == 2


def test_step_reinsert_and_fold_updates_heap_vectors():
    runtime = _runtime()
    runtime.prime_timestep_zero(
        layer_index=0,
        vectors=torch.randn(3, 8),
        scores=torch.tensor([0.9, 0.8, 0.1]),
        token_positions=torch.tensor([0, 1, 2]),
        head_indices=torch.tensor([0, 0, 1]),
        time_index=0,
    )
    popped = runtime.pop_top_k(layer_index=0)
    embedding = EmbeddingAutoencoder(hidden_size=8, latent_size=8, max_slots=4)
    relevancy = RelevancyTransformer(hidden_size=8, num_layers=1, num_heads=2)
    artifacts = runtime.step_with_reinsert_and_fold(
        layer_index=0,
        popped_entries=popped,
        transformed_popped_vectors=torch.randn(len(popped), 8),
        new_vectors=torch.randn(3, 8),
        new_scores=torch.tensor([0.7, 0.6, 0.5]),
        new_token_positions=torch.tensor([3, 4, 5]),
        new_head_indices=torch.tensor([0, 1, 0]),
        time_index=1,
        embedding_model=embedding,
        relevancy_model=relevancy,
    )
    new_heap = [entry.vector for entry in runtime.export_heap_entries(layer_index=0)]
    assert len(new_heap) <= runtime.config.model.max_heap_size
    assert artifacts.summary_embedding is not None
