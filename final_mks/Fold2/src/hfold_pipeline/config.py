from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .utils.io import load_yaml


ATTENTION_TYPE_ALIASES = {
    "full": "full",
    "self": "full",
    "self_attention": "full",
    "self-attention": "full",
    "sliding": "sliding_window",
    "sliding_attention": "sliding_window",
    "sliding-attention": "sliding_window",
    "sliding_window": "sliding_window",
    "sliding-window": "sliding_window",
    "hfold": "hfold",
    "hfold_attention": "hfold",
    "hfold-attention": "hfold",
}


def normalize_attention_type(attention_type: str) -> str:
    normalized = attention_type.strip().lower().replace(" ", "_")
    try:
        return ATTENTION_TYPE_ALIASES[normalized]
    except KeyError as exc:
        supported = "', '".join(sorted(ATTENTION_TYPE_ALIASES))
        raise ValueError(
            "attention.attention_type must be one of the supported names or aliases: "
            f"'{supported}'."
        ) from exc


@dataclass
class HFoldConfig:
    heap_size: int = 64
    top_q: int = 8
    pop_e: int = 8
    fold_hidden_size: int | None = None
    notes: str = ""


@dataclass
class ModelConfig:
    model_name: str = "EleutherAI/pythia-160m"
    tokenizer_name: str | None = None
    cache_dir: str = "artifacts/hf_cache"
    trust_remote_code: bool = False
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "eager"
    gradient_checkpointing: bool = True
    compile_model: bool = False


@dataclass
class AttentionConfig:
    attention_type: str = "full"
    window_size: int = 512
    allow_hfold_fallback: bool = False
    hfold_backend: str | None = None
    hfold: HFoldConfig = field(default_factory=HFoldConfig)


@dataclass
class DataConfig:
    dataset_name: str = "emozilla/pg19"
    dataset_config_name: str | None = None
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"
    text_column: str = "text"
    preprocessing_num_workers: int = 4
    block_size: int = 1024
    processed_dataset_dir: str = "artifacts/processed/pg19_block1024"
    overwrite_cache: bool = False
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    max_test_samples: int | None = None


@dataclass
class TrainingConfig:
    output_dir: str = "artifacts/runs/default"
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_train_epochs: float = 1.0
    max_steps: int = -1
    learning_rate: float = 5e-5
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    log_interval: int = 10
    eval_interval: int = 200
    save_interval: int = 200
    max_checkpoints: int = 3
    resume_from_checkpoint: str | None = None
    bf16: bool = True
    fp16: bool = False
    dataloader_num_workers: int = 2
    pin_memory: bool = True


@dataclass
class RuntimeConfig:
    seed: int = 42
    device: str = "auto"


@dataclass
class BenchmarkConfig:
    sequence_lengths: list[int] = field(default_factory=lambda: [512, 1024, 2048])
    max_eval_batches: int = 100
    profile_memory: bool = True
    profile_throughput: bool = True


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    def validate(self) -> None:
        self.attention.attention_type = normalize_attention_type(self.attention.attention_type)
        if self.attention.hfold_backend in {
            "",
            "your_package.your_module:build_hfold_backend",
        }:
            self.attention.hfold_backend = None
        if self.attention.window_size <= 0:
            raise ValueError("attention.window_size must be positive.")
        if self.attention.hfold.heap_size < 0:
            raise ValueError("attention.hfold.heap_size must be non-negative.")
        if self.attention.hfold.top_q < 0:
            raise ValueError("attention.hfold.top_q must be non-negative.")
        if self.attention.hfold.pop_e < 0:
            raise ValueError("attention.hfold.pop_e must be non-negative.")
        if self.data.block_size <= 0:
            raise ValueError("data.block_size must be positive.")
        if self.training.per_device_batch_size <= 0:
            raise ValueError("training.per_device_batch_size must be positive.")
        if self.training.gradient_accumulation_steps <= 0:
            raise ValueError("training.gradient_accumulation_steps must be positive.")
        if self.training.bf16 and self.training.fp16:
            raise ValueError("Choose at most one of training.bf16 or training.fp16.")
        if self.attention.attention_type == "hfold" and self.attention.hfold.pop_e > self.attention.hfold.heap_size:
            raise ValueError("attention.hfold.pop_e cannot exceed attention.hfold.heap_size.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct_dataclass(cls: type[Any], raw: dict[str, Any] | None) -> Any:
    raw = raw or {}
    if cls is AttentionConfig and "hfold" in raw:
        raw = dict(raw)
        raw["hfold"] = _construct_dataclass(HFoldConfig, raw["hfold"])
    return cls(**raw)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    raw = load_yaml(path)
    config = ExperimentConfig(
        model=_construct_dataclass(ModelConfig, raw.get("model")),
        attention=_construct_dataclass(AttentionConfig, raw.get("attention")),
        data=_construct_dataclass(DataConfig, raw.get("data")),
        training=_construct_dataclass(TrainingConfig, raw.get("training")),
        runtime=_construct_dataclass(RuntimeConfig, raw.get("runtime")),
        benchmark=_construct_dataclass(BenchmarkConfig, raw.get("benchmark")),
    )
    config.validate()
    return config
