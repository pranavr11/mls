import torch

from hfold_pipeline.attention.masking import build_sliding_window_attention_mask


def test_sliding_window_mask_is_causal_and_local():
    hidden_states = torch.zeros(1, 4, 8, dtype=torch.float32)
    mask = build_sliding_window_attention_mask(hidden_states, window_size=2)
    masked = mask[0, 0]
    min_value = torch.finfo(mask.dtype).min

    assert masked[0, 0] == 0
    assert masked[0, 1] == min_value
    assert masked[3, 1] == min_value
    assert masked[3, 2] == 0
    assert masked[3, 3] == 0


def test_sliding_window_mask_respects_cached_prefix_length():
    hidden_states = torch.zeros(1, 2, 8, dtype=torch.float32)
    fake_layer_past = (torch.zeros(1, 1, 3, 8),)
    mask = build_sliding_window_attention_mask(
        hidden_states,
        window_size=2,
        layer_past=fake_layer_past,
    )
    masked = mask[0, 0]
    min_value = torch.finfo(mask.dtype).min

    assert masked.shape == (2, 5)
    assert masked[0, 2] == 0
    assert masked[0, 3] == 0
    assert masked[0, 1] == min_value
    assert masked[0, 4] == min_value
    assert masked[1, 3] == 0
    assert masked[1, 4] == 0
