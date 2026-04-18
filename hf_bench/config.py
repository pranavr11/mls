from dataclasses import dataclass, asdict
from typing import List


@dataclass
class ExperimentConfig:
    models: List[str]
    datasets: List[str]
    scrolls_task: str
    seeds: List[int]
    results_root: str
    cache_dir: str
    epochs: int
    train_batch_size: int
    eval_batch_size: int
    learning_rate: float
    weight_decay: float
    max_train_steps: int
    eval_every_steps: int
    block_size: int
    eval_stride: int
    warmup_ratio: float
    grad_accum_steps: int
    max_grad_norm: float
    num_workers: int
    use_bf16: bool

    def to_dict(self):
        return asdict(self)


def default_config() -> ExperimentConfig:
    return ExperimentConfig(
        models=["EleutherAI/pythia-160m", "gpt2"],
        datasets=["pg19", "scrolls"],
        scrolls_task="gov_report",
        seeds=[13, 37, 73, 101],
        results_root="results",
        cache_dir=".cache",
        epochs=1,
        train_batch_size=1,
        eval_batch_size=1,
        learning_rate=2e-5,
        weight_decay=0.01,
        max_train_steps=300,
        eval_every_steps=50,
        block_size=1024,
        eval_stride=512,
        warmup_ratio=0.05,
        grad_accum_steps=1,
        max_grad_norm=1.0,
        num_workers=0,
        use_bf16=True,
    )
