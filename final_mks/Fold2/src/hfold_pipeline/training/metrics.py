from __future__ import annotations

import math
from typing import Any

import torch


def compute_perplexity(loss: float) -> float:
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def count_tokens_in_batch(batch: dict[str, torch.Tensor]) -> int:
    if "labels" in batch:
        return int(batch["labels"].ne(-100).sum().item())
    if "attention_mask" in batch:
        return int(batch["attention_mask"].sum().item())
    return int(batch["input_ids"].numel())


def get_peak_memory_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024**2)


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def maybe_item(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value

