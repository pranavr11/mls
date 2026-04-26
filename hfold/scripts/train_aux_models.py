"""Train embedding + relevancy aux models on real extracted shards.

Example:

    python -m hfold.scripts.train_aux_models \
      --extracted-dirs ./data/extracted/pythia ./data/extracted/gpt2 \
      --backbone-dims pythia=256 gpt2=768 \
      --output-dir ./checkpoints/aux \
      --epochs 3
"""
from __future__ import annotations

import argparse
import os

import torch

from hfold.config.schema import HFoldConfig, HFoldModelConfig, HFoldTrainingConfig
from hfold.data.hidden_state_dataset import HiddenStateShardDataset
from hfold.training.train_embedding import train_embedding_model
from hfold.training.train_relevancy import train_relevancy_model


def _parse_kv_list(values: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in values:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"expected key=value, got {item!r}")
        k, v = item.split("=", 1)
        out[k] = int(v)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extracted-dirs", nargs="+", required=True, help="One or more shard directories.")
    parser.add_argument(
        "--backbone-dims",
        nargs="+",
        required=True,
        help="Per-backbone hidden sizes, e.g. pythia=256 gpt2=768",
    )
    parser.add_argument("--max-heap-size", type=int, default=16)
    parser.add_argument("--adapter-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    backbone_dims = _parse_kv_list(args.backbone_dims)

    # Use the largest hidden size as the canonical "model.hidden_size"; the
    # adapter registry handles per-backbone shapes individually.
    canonical_hidden = max(backbone_dims.values())

    config = HFoldConfig(
        model=HFoldModelConfig(
            hidden_size=canonical_hidden,
            num_heads=args.num_heads,
            max_heap_size=args.max_heap_size,
            top_w=args.max_heap_size,
            pop_k=args.max_heap_size,
            adapter_dim=args.adapter_dim,
        ),
        training=HFoldTrainingConfig(
            learning_rate=args.lr,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            seed=args.seed,
        ),
    )

    dataset = HiddenStateShardDataset(args.extracted_dirs)
    print({"num_shards_total": len(dataset)})

    emb_artifacts = train_embedding_model(
        config=config,
        dataset=dataset,
        backbone_dims=backbone_dims,
    )
    rel_artifacts = train_relevancy_model(
        config=config,
        dataset=dataset,
        embedding_model=emb_artifacts.model,
        adapters=emb_artifacts.adapters,
    )

    torch.save(emb_artifacts.model.state_dict(), os.path.join(args.output_dir, "embedding_autoencoder.pt"))
    torch.save(emb_artifacts.adapters.state_dict(), os.path.join(args.output_dir, "adapters.pt"))
    torch.save(rel_artifacts.model.state_dict(), os.path.join(args.output_dir, "relevancy_transformer.pt"))
    print(
        {
            "embedding_loss": emb_artifacts.final_loss,
            "relevancy_loss": rel_artifacts.final_loss,
            "output_dir": args.output_dir,
        }
    )


if __name__ == "__main__":
    main()
