"""
Configuration and utilities for HFOLD models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class HFoldConfig:
    """Configuration for HFOLD models"""
    
    # Model architecture
    vocab_size: int = 50257  # GPT-2 vocab size
    d_model: int = 768       # Model dimension
    n_heads: int = 12        # Number of attention heads
    n_layers: int = 12       # Number of transformer layers
    d_ff: int = 3072         # Feed-forward hidden dimension
    max_seq_len: int = 2048  # Maximum sequence length
    
    # HFOLD attention parameters
    window_size: int = 64       # k: sliding window size
    heap_size: int = 32         # s: max heap size (constant)
    q_topk: int = 16            # q: top-q keys to add to heap
    e_pop: int = 8              # e: top-e keys to pop from heap
    
    # Training
    dropout: float = 0.1
    activation: str = "gelu"
    layer_norm_eps: float = 1e-5
    
    def __post_init__(self):
        """Validation"""
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.e_pop <= self.heap_size, "e_pop cannot exceed heap_size"
        assert self.window_size > 0, "window_size must be positive"
        assert self.q_topk > 0, "q_topk must be positive"

        # Upper bound on candidates per HFold step once the window is full:
        # at most window_size prior/current positions in-window plus e_pop heap slots.
        # Early positions use a shorter window (length <= token_idx+1).
        self.effective_context = self.e_pop + self.window_size

        # Rough dot-product budget per token per layer for HFold attention (linear in n over the sequence).
        # Full self-attention is Θ(n²); HFold attention kernel per position is Θ(d_k · (window + e_pop)).
        self.theoretical_flops_per_token = (self.d_model * (self.window_size + self.e_pop)) / 1e9


@dataclass
class TrainingConfig:
    """Configuration for training"""
    
    batch_size: int = 32
    max_epochs: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    
    # Evaluation
    eval_steps: int = 500
    save_steps: int = 1000
    
    # Data
    data_path: str = "/tmp/pg19"
    output_dir: str = "./checkpoints"


class RMSNorm(nn.Module):
    """Root mean square normalization (scale-invariant per row)."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight


class FeedForward(nn.Module):
    """Feed-forward network"""
    
    def __init__(self, d_model: int, d_ff: int, activation: str = "gelu", dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
