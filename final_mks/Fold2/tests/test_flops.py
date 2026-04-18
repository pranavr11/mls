from hfold_pipeline.config import AttentionConfig
from hfold_pipeline.training.flops import estimate_attention_flops


def test_attention_flops_drop_for_sliding_window():
    full = estimate_attention_flops(
        attention_config=AttentionConfig(attention_type="full", window_size=8),
        num_hidden_layers=2,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        sequence_length=16,
        batch_size=1,
    )
    sliding = estimate_attention_flops(
        attention_config=AttentionConfig(attention_type="sliding_window", window_size=4),
        num_hidden_layers=2,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        sequence_length=16,
        batch_size=1,
    )

    assert sliding.attention_pairs_per_layer < full.attention_pairs_per_layer
    assert sliding.attention_backend_flops_total < full.attention_backend_flops_total

