from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HFoldModelConfig:
    hidden_size: int
    num_heads: int
    max_heap_size: int = 16
    top_w: int = 8
    pop_k: int = 8
    adapter_dim: int = 256
    embedding_latent_dim: int | None = None

    def validate(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")
        if self.max_heap_size < 0:
            raise ValueError("max_heap_size must be >= 0.")
        if self.top_w < 0:
            raise ValueError("top_w must be >= 0.")
        if self.pop_k < 0:
            raise ValueError("pop_k must be >= 0.")
        # Interpret out-of-range defaults as "use up to heap capacity".
        if self.pop_k > self.max_heap_size:
            self.pop_k = self.max_heap_size
        if self.top_w > 0 and self.max_heap_size == 0:
            self.top_w = 0
        if self.adapter_dim <= 0:
            raise ValueError("adapter_dim must be positive.")
        # Auto-resolve a true bottleneck by default.
        if self.embedding_latent_dim is None:
            self.embedding_latent_dim = max(1, self.adapter_dim // 2)
        if self.embedding_latent_dim <= 0:
            raise ValueError("embedding_latent_dim must be positive.")
        if self.adapter_dim > 1 and self.embedding_latent_dim >= self.adapter_dim:
            raise ValueError("embedding_latent_dim must be < adapter_dim for a true bottleneck.")


@dataclass
class HFoldTrainingConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 16
    num_epochs: int = 5
    max_steps: int | None = None
    cosine_loss_weight: float = 1.0
    mse_loss_weight: float = 0.25
    ranking_loss_weight: float = 0.15
    gradient_clip_norm: float = 1.0
    device: str = "auto"
    seed: int = 42

    def validate(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive.")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive when provided.")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive.")


@dataclass
class HFoldConfig:
    model: HFoldModelConfig
    training: HFoldTrainingConfig = field(default_factory=HFoldTrainingConfig)
    pad_token_strategy: str = "zero"
    backbone_adapters: tuple[str, ...] = ("pythia", "gpt2")

    def validate(self) -> None:
        self.model.validate()
        self.training.validate()
        if self.pad_token_strategy not in {"zero", "learned"}:
            raise ValueError("pad_token_strategy must be one of: zero, learned.")
        if not self.backbone_adapters:
            raise ValueError("backbone_adapters cannot be empty.")
