from __future__ import annotations

import argparse

from ..config import load_experiment_config
from ..training.trainer import train
from ..utils.logging import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Train Pythia-160M with a pluggable attention backend.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML.")
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
    train(config)


if __name__ == "__main__":
    main()
