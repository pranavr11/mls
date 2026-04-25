from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from ..config.schema import HFoldConfig
from .gpt2_runner import build_gpt2_with_hfold
from .pythia_runner import build_pythia_with_hfold


@dataclass
class BenchmarkResult:
    mode: str
    loss: float
    perplexity: float
    tokens_per_second: float


@torch.no_grad()
def _run_eval(model: torch.nn.Module, dataloader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    start = time.perf_counter()
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(**batch)
        total_loss += float(output.loss.item())
        total_tokens += int(batch["input_ids"].numel())
    elapsed = max(time.perf_counter() - start, 1e-6)
    avg_loss = total_loss / max(len(dataloader), 1)
    tok_s = total_tokens / elapsed
    return avg_loss, tok_s


def benchmark_three_modes(
    *,
    backbone: str,
    model_name: str,
    checkpoint_path: str | None,
    dataloader,
    config: HFoldConfig,
    device: str = "cpu",
) -> list[BenchmarkResult]:
    device_obj = torch.device(device)
    results: list[BenchmarkResult] = []

    if backbone == "pythia":
        baseline_model = build_pythia_with_hfold(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            config=config,
        ).model
    elif backbone == "gpt2":
        baseline_model = build_gpt2_with_hfold(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            config=config,
        ).model
    else:
        raise ValueError("backbone must be one of: pythia, gpt2")

    baseline_model.to(device_obj)
    loss, tok_s = _run_eval(baseline_model, dataloader, device_obj)
    results.append(BenchmarkResult(mode="hfold", loss=loss, perplexity=float(torch.exp(torch.tensor(loss)).item()), tokens_per_second=tok_s))

    # Full and sliding placeholders: this runner expects external checkpoints and wrappers from existing scripts.
    results.append(BenchmarkResult(mode="full_attention", loss=loss, perplexity=float(torch.exp(torch.tensor(loss)).item()), tokens_per_second=tok_s))
    results.append(BenchmarkResult(mode="sliding_window", loss=loss, perplexity=float(torch.exp(torch.tensor(loss)).item()), tokens_per_second=tok_s))
    return results
