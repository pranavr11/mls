import torch
from torch import nn

from hfold.config.schema import HFoldConfig, HFoldModelConfig
from hfold.inference.attention_patch import patch_pythia_model_attention
from hfold.inference.hfold_runtime import HFoldRuntime
from hfold.models.embedding_autoencoder import EmbeddingAutoencoder
from hfold.models.relevancy_transformer import RelevancyTransformer


class DummyAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.query_key_value = nn.Linear(hidden_size, hidden_size * 3)
        self.dense = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, *args, **kwargs):
        del args, kwargs
        output = self.dense(hidden_states)
        b, s, _ = hidden_states.shape
        attn = torch.softmax(torch.randn(b, 1, s, s), dim=-1)
        return output, None, attn


class DummyPythia(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.layer0 = DummyAttention(hidden_size)
        self.layer1 = DummyAttention(hidden_size)

    def forward(self, x):
        x, _, _ = self.layer0(x)
        x, _, _ = self.layer1(x)
        return x


def test_patch_pythia_attention_smoke():
    config = HFoldConfig(model=HFoldModelConfig(hidden_size=8, num_heads=2, max_heap_size=4))
    runtime = HFoldRuntime(config)
    model = DummyPythia(hidden_size=8)
    embed = EmbeddingAutoencoder(hidden_size=8, latent_size=8, max_slots=4)
    rel = RelevancyTransformer(hidden_size=8, num_layers=1, num_heads=2)
    patch_pythia_model_attention(model, runtime, embed, rel)
    out = model(torch.randn(1, 6, 8))
    assert out.shape == (1, 6, 8)
