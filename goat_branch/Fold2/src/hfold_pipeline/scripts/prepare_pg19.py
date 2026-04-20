from __future__ import annotations

import argparse
import logging

from ..config import load_experiment_config
from ..data.pg19 import load_or_prepare_pg19
from ..modeling.pythia import load_pythia_tokenizer
from ..utils.logging import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess PG-19 for Pythia fine-tuning.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML.")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()
    config = load_experiment_config(args.config)
    tokenizer = load_pythia_tokenizer(config)
    dataset_dict = load_or_prepare_pg19(tokenizer, config.data)
    logging.getLogger(__name__).info("Prepared splits: %s", list(dataset_dict.keys()))


if __name__ == "__main__":
    main()
