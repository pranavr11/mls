from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path

from ..config import load_experiment_config
from ..training.flops import estimate_attention_flops
from ..training.trainer import evaluate_checkpoint
from ..utils.io import save_json
from ..utils.logging import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark a trained checkpoint across sequence lengths.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint directory to load.")
    parser.add_argument(
        "--attention-type",
        default=None,
        help="Optional attention override. Supports hfold, full/self_attention, and sliding_window/sliding_attention.",
    )
    return parser.parse_args()


def main():
    from transformers import AutoConfig

    args = parse_args()
    setup_logging()
    base_config = load_experiment_config(args.config)
    if args.attention_type is not None:
        base_config.attention.attention_type = args.attention_type
        base_config.validate()
    model_config = AutoConfig.from_pretrained(
        base_config.model.model_name,
        cache_dir=base_config.model.cache_dir,
        trust_remote_code=base_config.model.trust_remote_code,
    )

    results = {}
    for sequence_length in base_config.benchmark.sequence_lengths:
        config = copy.deepcopy(base_config)
        config.data.block_size = sequence_length
        config.data.processed_dataset_dir = (
            f"artifacts/processed/pg19_block{sequence_length}"
        )
        metrics = evaluate_checkpoint(
            config,
            checkpoint_path=args.checkpoint,
            split=config.data.validation_split,
            max_batches=config.benchmark.max_eval_batches,
            profile_throughput=config.benchmark.profile_throughput,
            profile_memory=config.benchmark.profile_memory,
        )
        flop_estimate = estimate_attention_flops(
            attention_config=config.attention,
            num_hidden_layers=model_config.num_hidden_layers,
            hidden_size=model_config.hidden_size,
            intermediate_size=model_config.intermediate_size,
            num_attention_heads=model_config.num_attention_heads,
            sequence_length=sequence_length,
            batch_size=config.training.per_device_batch_size,
        )
        results[str(sequence_length)] = {
            **metrics,
            "estimated_attention_pairs_per_layer": flop_estimate.attention_pairs_per_layer,
            "estimated_attention_backend_flops_total": flop_estimate.attention_backend_flops_total,
            "estimated_total_transformer_forward_flops": flop_estimate.total_transformer_flops_forward,
        }
        logging.getLogger(__name__).info(
            "Benchmark length=%d metrics=%s",
            sequence_length,
            results[str(sequence_length)],
        )

    output_path = Path(base_config.training.output_dir) / "benchmark.json"
    save_json(results, output_path)


if __name__ == "__main__":
    main()
