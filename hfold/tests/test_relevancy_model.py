import torch

from hfold.models.relevancy_transformer import RelevancyTransformer


def test_relevancy_outputs_heap_scores():
    model = RelevancyTransformer(hidden_size=12, num_layers=1, num_heads=3)
    summary = torch.randn(2, 12)
    heap = torch.randn(2, 5, 12)
    scores = model(summary, heap)
    assert scores.shape == (2, 5)
