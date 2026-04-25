from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from hfold.config.schema import HFoldConfig, HFoldModelConfig, HFoldTrainingConfig
from hfold.data.hidden_state_dataset import SyntheticHiddenStateDataset
from hfold.integration.benchmark_runner import benchmark_three_modes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["pythia", "gpt2"], required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--hidden-size", type=int, default=768)
    return parser.parse_args()


def _dummy_lm_collate(_samples):
    return {
        "input_ids": torch.ones(1, 16, dtype=torch.long),
        "attention_mask": torch.ones(1, 16, dtype=torch.long),
        "labels": torch.ones(1, 16, dtype=torch.long),
    }


def main():
    args = parse_args()
    config = HFoldConfig(model=HFoldModelConfig(hidden_size=args.hidden_size, num_heads=12), training=HFoldTrainingConfig())
    dataset = SyntheticHiddenStateDataset(size=4, backbone=args.backbone, hidden_size=args.hidden_size, max_heap_size=config.model.max_heap_size)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=_dummy_lm_collate)
    results = benchmark_three_modes(
        backbone=args.backbone,
        model_name=args.model_name,
        checkpoint_path=args.checkpoint,
        dataloader=dataloader,
        config=config,
    )
    print([result.__dict__ for result in results])


if __name__ == "__main__":
    main()
