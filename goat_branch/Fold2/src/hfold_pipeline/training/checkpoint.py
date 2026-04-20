from __future__ import annotations

import logging
from pathlib import Path
import shutil
from typing import Any

import torch

from ..utils.io import ensure_dir, save_json

logger = logging.getLogger(__name__)


def _unwrap_model(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def save_checkpoint(
    *,
    checkpoint_dir: str | Path,
    model,
    tokenizer,
    optimizer,
    scheduler,
    scaler,
    experiment_config,
    trainer_state: dict[str, Any],
) -> Path:
    checkpoint_dir = ensure_dir(checkpoint_dir)
    model_to_save = _unwrap_model(model)

    torch.save(model_to_save.state_dict(), checkpoint_dir / "model_state.pt")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "trainer_state": trainer_state,
        },
        checkpoint_dir / "trainer_state.pt",
    )

    model_to_save.config.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    save_json(experiment_config.to_dict(), checkpoint_dir / "experiment_config.json")
    logger.info("Saved checkpoint to %s", checkpoint_dir)
    return checkpoint_dir


def load_checkpoint(
    *,
    checkpoint_dir: str | Path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    model_state = torch.load(checkpoint_dir / "model_state.pt", map_location=device)
    _unwrap_model(model).load_state_dict(model_state)

    trainer_state_path = checkpoint_dir / "trainer_state.pt"
    if not trainer_state_path.exists():
        return {}

    payload = torch.load(trainer_state_path, map_location=device)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    return payload.get("trainer_state", {})


def cleanup_old_checkpoints(output_dir: str | Path, max_checkpoints: int) -> None:
    output_dir = Path(output_dir)
    checkpoints = sorted(
        [p for p in output_dir.glob("checkpoint-*") if p.is_dir()],
        key=lambda path: path.stat().st_mtime,
    )
    while len(checkpoints) > max_checkpoints:
        doomed = checkpoints.pop(0)
        logger.info("Removing old checkpoint %s", doomed)
        shutil.rmtree(doomed)


def find_latest_checkpoint(output_dir: str | Path) -> Path | None:
    output_dir = Path(output_dir)
    checkpoints = sorted(
        [p for p in output_dir.glob("checkpoint-*") if p.is_dir()],
        key=lambda path: path.stat().st_mtime,
    )
    return checkpoints[-1] if checkpoints else None
