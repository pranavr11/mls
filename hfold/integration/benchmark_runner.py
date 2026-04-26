from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

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
    total_batches = 0
    start = time.perf_counter()
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(**batch)
        total_loss += float(output.loss.item())
        total_batches += 1
        total_tokens += int(batch["input_ids"].numel())
    elapsed = max(time.perf_counter() - start, 1e-6)
    avg_loss = total_loss / max(total_batches, 1)
    tok_s = total_tokens / elapsed
    return avg_loss, tok_s


def _to_result(mode: str, loss: float, tok_s: float) -> BenchmarkResult:
    return BenchmarkResult(
        mode=mode,
        loss=loss,
        perplexity=float(math.exp(loss)) if loss < 50.0 else float("inf"),
        tokens_per_second=tok_s,
    )


class _SlidingWindowMaskWrapper(torch.nn.Module):
    """Inline sliding-window mask wrapper, mirroring `new_fine_tune.py` semantics."""

    def __init__(self, original_attention: torch.nn.Module, window_size: int) -> None:
        super().__init__()
        self.original_attention = original_attention
        self.window_size = window_size

    def forward(self, hidden_states, *args, **kwargs):
        if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
            mask = kwargs["attention_mask"]
            if mask.dim() == 4:
                tgt_len = mask.shape[-2]
                src_len = mask.shape[-1]
                idx_tgt = torch.arange(src_len - tgt_len, src_len, device=mask.device).unsqueeze(1)
                idx_src = torch.arange(src_len, device=mask.device).unsqueeze(0)
                out_of_window = (idx_tgt - idx_src) >= self.window_size
                modified = mask.clone()
                if modified.dtype == torch.bool:
                    modified = modified.masked_fill(out_of_window, False)
                else:
                    min_val = torch.finfo(modified.dtype).min
                    modified = modified.masked_fill(out_of_window, min_val)
                kwargs["attention_mask"] = modified
        return self.original_attention(hidden_states, *args, **kwargs)


def _apply_sliding_window(model: torch.nn.Module, window_size: int) -> torch.nn.Module:
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        for layer in model.gpt_neox.layers:
            layer.attention = _SlidingWindowMaskWrapper(layer.attention, window_size)
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        for layer in model.transformer.h:
            layer.attn = _SlidingWindowMaskWrapper(layer.attn, window_size)
    else:
        raise ValueError("Unsupported model architecture for sliding-window benchmark.")
    return model


def benchmark_three_modes(
    *,
    backbone: str,
    model_name: str,
    checkpoint_path: str | None,
    dataloader,
    config: HFoldConfig,
    device: str = "cpu",
    sliding_window_size: int = 256,
    full_model_factory: Callable[[], torch.nn.Module] | None = None,
    sliding_model_factory: Callable[[], torch.nn.Module] | None = None,
) -> list[BenchmarkResult]:
    """
    Benchmark full / sliding / hfold on the SAME dataloader.

    Each mode uses a freshly built model (or caller-supplied factory). Metrics are
    computed independently per mode; no placeholder duplication.
    """
    device_obj = torch.device(device)
    results: list[BenchmarkResult] = []

    if backbone not in {"pythia", "gpt2"}:
        raise ValueError("backbone must be one of: pythia, gpt2")

    def _default_full_factory() -> torch.nn.Module:
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(checkpoint_path or model_name)

    def _default_sliding_factory() -> torch.nn.Module:
        model = (full_model_factory or _default_full_factory)()
        return _apply_sliding_window(model, sliding_window_size)

    def _build_hfold() -> torch.nn.Module:
        if backbone == "pythia":
            return build_pythia_with_hfold(model_name=model_name, checkpoint_path=checkpoint_path, config=config).model
        return build_gpt2_with_hfold(model_name=model_name, checkpoint_path=checkpoint_path, config=config).model

    full_model = (full_model_factory or _default_full_factory)().to(device_obj)
    full_loss, full_tok_s = _run_eval(full_model, dataloader, device_obj)
    results.append(_to_result("full_attention", full_loss, full_tok_s))

    sliding_model = (sliding_model_factory or _default_sliding_factory)().to(device_obj)
    sliding_loss, sliding_tok_s = _run_eval(sliding_model, dataloader, device_obj)
    results.append(_to_result("sliding_window", sliding_loss, sliding_tok_s))

    hfold_model = _build_hfold().to(device_obj)
    hfold_loss, hfold_tok_s = _run_eval(hfold_model, dataloader, device_obj)
    results.append(_to_result("hfold", hfold_loss, hfold_tok_s))

    return results
