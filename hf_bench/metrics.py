from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class RuntimeAverages:
    train_step_time_s: float
    forward_time_s: float
    tokens_per_sec: float
    peak_memory_mb: float
    flops_per_step: float


def estimate_decoder_flops(model, batch_size: int, seq_len: int) -> float:
    cfg = model.config
    layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer")
    hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd")
    vocab = getattr(cfg, "vocab_size")

    bsz_tokens = batch_size * seq_len
    attn = 4.0 * bsz_tokens * seq_len * hidden * layers
    mlp = 16.0 * bsz_tokens * hidden * hidden * layers
    logits = 2.0 * bsz_tokens * hidden * vocab
    return attn + mlp + logits


def peak_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def to_serializable(metrics: RuntimeAverages) -> Dict[str, float]:
    return {
        "train_step_time_s": metrics.train_step_time_s,
        "forward_time_s": metrics.forward_time_s,
        "tokens_per_sec": metrics.tokens_per_sec,
        "peak_memory_mb": metrics.peak_memory_mb,
        "flops_per_step": metrics.flops_per_step,
    }
