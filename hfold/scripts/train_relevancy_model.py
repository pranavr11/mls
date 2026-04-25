from __future__ import annotations

import argparse

from hfold.config.schema import HFoldConfig, HFoldModelConfig, HFoldTrainingConfig
from hfold.data.hidden_state_dataset import SyntheticHiddenStateDataset
from hfold.training.train_embedding import train_embedding_model
from hfold.training.train_relevancy import train_relevancy_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--heap-size", type=int, default=16)
    parser.add_argument("--dataset-size", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    config = HFoldConfig(
        model=HFoldModelConfig(hidden_size=args.hidden_size, num_heads=12, max_heap_size=args.heap_size),
        training=HFoldTrainingConfig(),
    )
    dataset = SyntheticHiddenStateDataset(
        size=args.dataset_size,
        backbone="pythia",
        hidden_size=args.hidden_size,
        max_heap_size=args.heap_size,
    )
    emb = train_embedding_model(config=config, dataset=dataset, backbone_dims={"pythia": args.hidden_size, "gpt2": args.hidden_size})
    rel = train_relevancy_model(config=config, dataset=dataset, embedding_model=emb.model, adapters=emb.adapters)
    print({"embedding_loss": emb.final_loss, "relevancy_loss": rel.final_loss})


if __name__ == "__main__":
    main()
