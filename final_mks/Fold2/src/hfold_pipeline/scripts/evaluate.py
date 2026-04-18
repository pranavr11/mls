from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..config import load_experiment_config
from ..training.trainer import evaluate_checkpoint
from ..utils.io import save_json
from ..utils.logging import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint directory to load.")
    parser.add_argument("--split", default=None, help="Dataset split to evaluate.")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional cap on eval batches.")
    parser.add_argument(
        "--attention-type",
        default=None,
        help="Optional attention override. Supports hfold, full/self_attention, and sliding_window/sliding_attention.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()
    config = load_experiment_config(args.config)
    if args.attention_type is not None:
        config.attention.attention_type = args.attention_type
        config.validate()
    metrics = evaluate_checkpoint(
        config,
        checkpoint_path=args.checkpoint,
        split=args.split,
        max_batches=args.max_batches,
    )
    output_path = Path(config.training.output_dir) / "evaluation.json"
    save_json(metrics, output_path)
    logging.getLogger(__name__).info("Evaluation metrics: %s", metrics)


if __name__ == "__main__":
    main()
