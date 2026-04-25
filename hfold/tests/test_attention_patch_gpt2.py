import torch
from torch import nn

from hfold.config.schema import HFoldConfig, HFoldModelConfig
from hfold.inference.attention_patch import patch_gpt2_model_attention
from hfold.inference.hfold_runtime import HFoldRuntime
from hfold.models.embedding_autoencoder import EmbeddingAutoencoder
from hfold.models.relevancy_transformer import RelevancyTransformer


class DummyGPT2Attention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, *args, **kwargs):
        del args, kwargs
        out = self.proj(hidden_states)
        b, s, _ = hidden_states.shape
        attn = torch.softmax(torch.randn(b, 1, s, s), dim=-1)
        return out, None, attn


class DummyBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = DummyGPT2Attention(hidden_size)

    def forward(self, x):
        x, _, _ = self.attn(x)
        return x


class DummyGPT2Model(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([DummyBlock(hidden_size), DummyBlock(hidden_size)])

    def forward(self, x):
        for block in self.transformer.h:
            x = block(x)
        return x


def test_patch_gpt2_attention_smoke():
    config = HFoldConfig(model=HFoldModelConfig(hidden_size=8, num_heads=2, max_heap_size=4))
    runtime = HFoldRuntime(config)
    model = DummyGPT2Model(hidden_size=8)
    embed = EmbeddingAutoencoder(hidden_size=8, latent_size=8, max_slots=4)
    rel = RelevancyTransformer(hidden_size=8, num_layers=1, num_heads=2)
    patch_gpt2_model_attention(model, runtime, embed, rel)
    out = model(torch.randn(1, 6, 8))
    assert out.shape == (1, 6, 8)
