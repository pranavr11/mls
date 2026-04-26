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


def test_decoder_emits_distinct_slots():
    """Reconstructions must differ across slots; otherwise we cannot model
    slot-specific evicted vectors as required by the algorithm spec.
    """
    torch.manual_seed(0)
    model = EmbeddingAutoencoder(hidden_size=8, latent_size=8, max_slots=4)
    x = torch.arange(4 * 8, dtype=torch.float32).reshape(1, 4, 8) * 0.05
    reconstructed, _ = model(x)
    pairwise_diffs = []
    for i in range(reconstructed.size(1)):
        for j in range(i + 1, reconstructed.size(1)):
            pairwise_diffs.append(float((reconstructed[0, i] - reconstructed[0, j]).abs().max().item()))
    assert min(pairwise_diffs) > 1e-6, (
        "decoder must produce distinct reconstructions per slot"
    )
