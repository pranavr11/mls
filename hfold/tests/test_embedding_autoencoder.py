import torch

from hfold.models.embedding_autoencoder import EmbeddingAutoencoder


def test_autoencoder_shapes():
    model = EmbeddingAutoencoder(hidden_size=16, latent_size=12, max_slots=4)
    x = torch.randn(2, 4, 16)
    reconstructed, summary = model(x)
    assert reconstructed.shape == x.shape
    assert summary.shape == (2, 12)


def test_encode_summary_masked():
    model = EmbeddingAutoencoder(hidden_size=8, latent_size=8, max_slots=4)
    x = torch.randn(1, 4, 8)
    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
    summary = model.encode_summary(x, padding_mask=mask)
    assert summary.shape == (1, 8)
