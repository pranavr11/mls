from __future__ import annotations

from dataclasses import dataclass

from ..config import AttentionConfig, normalize_attention_type


@dataclass
class AttentionFlopEstimate:
    attention_pairs_per_layer: int
    attention_mix_flops_per_layer: int
    attention_softmax_flops_per_layer: int
    attention_backend_flops_per_layer: int
    attention_backend_flops_total: int
    projection_flops_per_layer: int
    mlp_flops_per_layer: int
    total_transformer_flops_per_layer: int
    total_transformer_flops_forward: int


def estimate_attention_flops(
    *,
    attention_config: AttentionConfig,
    num_hidden_layers: int,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    sequence_length: int,
    batch_size: int,
) -> AttentionFlopEstimate:
    pairs = _estimate_attention_pairs(
        seq_len=sequence_length,
        attention_config=attention_config,
    )

    attention_mix_flops_per_layer = 4 * batch_size * hidden_size * pairs
    attention_softmax_flops_per_layer = 3 * batch_size * num_attention_heads * pairs
    attention_backend_flops_per_layer = (
        attention_mix_flops_per_layer + attention_softmax_flops_per_layer
    )
    projection_flops_per_layer = 8 * batch_size * sequence_length * hidden_size * hidden_size
    mlp_flops_per_layer = 4 * batch_size * sequence_length * hidden_size * intermediate_size
    total_transformer_flops_per_layer = (
        projection_flops_per_layer + mlp_flops_per_layer + attention_backend_flops_per_layer
    )

    return AttentionFlopEstimate(
        attention_pairs_per_layer=pairs,
        attention_mix_flops_per_layer=attention_mix_flops_per_layer,
        attention_softmax_flops_per_layer=attention_softmax_flops_per_layer,
        attention_backend_flops_per_layer=attention_backend_flops_per_layer,
        attention_backend_flops_total=attention_backend_flops_per_layer * num_hidden_layers,
        projection_flops_per_layer=projection_flops_per_layer,
        mlp_flops_per_layer=mlp_flops_per_layer,
        total_transformer_flops_per_layer=total_transformer_flops_per_layer,
        total_transformer_flops_forward=total_transformer_flops_per_layer * num_hidden_layers,
    )


def _estimate_attention_pairs(*, seq_len: int, attention_config: AttentionConfig) -> int:
    attention_type = normalize_attention_type(attention_config.attention_type)

    if attention_type == "full":
        return seq_len * (seq_len + 1) // 2

    window = max(1, attention_config.window_size)
    if seq_len <= window:
        local_pairs = seq_len * (seq_len + 1) // 2
    else:
        local_pairs = window * (window + 1) // 2 + (seq_len - window) * window

    if attention_type == "sliding_window":
        return local_pairs

    if attention_type == "hfold":
        return local_pairs + seq_len * attention_config.hfold.pop_e

    raise ValueError(f"Unsupported attention type: {attention_type}")
