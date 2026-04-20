import torch
from torch import nn

from hfold_pipeline.config import AttentionConfig
from hfold_pipeline.modeling.pythia import patch_attention_module


class DummyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_key_value = nn.Linear(4, 12)
        self.dense = nn.Linear(4, 4)
        self.last_attention_mask = None

    def forward(self, hidden_states, attention_mask=None, layer_past=None):
        self.last_attention_mask = attention_mask
        return hidden_states


def test_patch_attention_module_rewrites_attention_mask():
    module = DummyAttention()
    config = AttentionConfig(attention_type="sliding_window", window_size=2)
    patch_attention_module(
        module=module,
        layer_index=0,
        attention_config=config,
        hfold_backend=None,
    )

    hidden_states = torch.zeros(1, 4, 4)
    module(hidden_states)

    assert module.last_attention_mask is not None
    assert module.last_attention_mask.shape[-2:] == (4, 4)


class DummyHFoldAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_key_value = nn.Linear(4, 12, bias=False)
        self.dense = nn.Linear(4, 4, bias=False)
        self.num_attention_heads = 1

    def forward(self, hidden_states, attention_mask=None, layer_past=None, use_cache=False, output_attentions=False):
        del attention_mask, layer_past, use_cache, output_attentions
        return hidden_states, None


def test_patch_attention_module_builds_native_hfold_backend():
    module = DummyHFoldAttention()
    config = AttentionConfig(attention_type="hfold", window_size=2)
    patch_attention_module(
        module=module,
        layer_index=0,
        attention_config=config,
        hfold_backend=None,
    )

    assert hasattr(module, "hfold_native_core")

    hidden_states = torch.randn(1, 4, 4)
    attn_output, present = module(hidden_states, attention_mask=torch.ones(1, 4, dtype=torch.long))

    assert attn_output.shape == hidden_states.shape
    assert present is None


def test_native_hfold_patch_remains_causal():
    torch.manual_seed(7)
    module = DummyHFoldAttention().eval()
    config = AttentionConfig(attention_type="hfold", window_size=3)
    patch_attention_module(
        module=module,
        layer_index=0,
        attention_config=config,
        hfold_backend=None,
    )

    hidden_states = torch.randn(1, 6, 4)
    altered = hidden_states.clone()
    altered[:, -1, :] = altered[:, -1, :] + 50.0

    output_a, _ = module(hidden_states, attention_mask=torch.ones(1, 6, dtype=torch.long))
    output_b, _ = module(altered, attention_mask=torch.ones(1, 6, dtype=torch.long))

    torch.testing.assert_close(output_a[:, :-1, :], output_b[:, :-1, :], rtol=1e-5, atol=1e-5)
