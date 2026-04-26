"""CLI to extract real hidden-state shards from a fine-tuned backbone.

Example:

    python -m hfold.scripts.extract_hidden_states \
      --backbone pythia \
      --model-name EleutherAI/pythia-31m \
      --checkpoint-dir ./checkpoints/pythia_full_finetuning \
      --dataset wikitext \
      --output-dir ./data/extracted/pythia
"""
from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from hfold.data.extract_hidden_states import ExtractionConfig, extract_to_shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["pythia", "gpt2"], required=True)
    parser.add_argument("--model-name", required=True, help="HF hub name (used as fallback if no checkpoint).")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Local fine-tuned checkpoint directory; falls back to --model-name if omitted.",
    )
    parser.add_argument("--dataset", choices=["wikitext"], default="wikitext")
    parser.add_argument("--wikitext-config", default="wikitext-103-raw-v1")
    parser.add_argument("--cache-dir", default="./data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-len", type=int, default=512)
    parser.add_argument("--max-heap-size", type=int, default=16)
    parser.add_argument("--num-anchors-per-chunk", type=int, default=4)
    parser.add_argument("--max-chunks", type=int, default=128)
    parser.add_argument("--samples-per-shard", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def _build_dataloader(args: argparse.Namespace, tokenizer) -> DataLoader:
    from datasets import load_dataset

    if args.dataset == "wikitext":
        raw = load_dataset("wikitext", args.wikitext_config, cache_dir=args.cache_dir)
    else:
        raise ValueError(f"unsupported dataset: {args.dataset}")
    split = "train" if "train" in raw else next(iter(raw.keys()))

    def tokenize(batch):
        return tokenizer(batch["text"], add_special_tokens=False)

    tokenized = raw[split].map(tokenize, batched=True, remove_columns=raw[split].column_names)

    chunk_len = args.chunk_len

    def group(batch):
        concat: list[int] = []
        for ids in batch["input_ids"]:
            concat.extend(ids)
        n = (len(concat) // chunk_len) * chunk_len
        chunks = [concat[i : i + chunk_len] for i in range(0, n, chunk_len)]
        return {"input_ids": chunks, "attention_mask": [[1] * chunk_len for _ in chunks]}

    grouped = tokenized.map(group, batched=True, remove_columns=tokenized.column_names)

    def collate(rows):
        ids = torch.tensor([row["input_ids"] for row in rows], dtype=torch.long)
        mask = torch.tensor([row["attention_mask"] for row in rows], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask}

    return DataLoader(grouped, batch_size=1, shuffle=False, collate_fn=collate)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = _resolve_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_path = args.checkpoint_dir if args.checkpoint_dir else args.model_name
    model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=args.cache_dir).to(device)

    dataloader = _build_dataloader(args, tokenizer)
    config = ExtractionConfig(
        backbone=args.backbone,
        chunk_len=args.chunk_len,
        max_heap_size=args.max_heap_size,
        num_anchors_per_chunk=args.num_anchors_per_chunk,
        seed=args.seed,
    )
    total = extract_to_shards(
        model=model,
        dataloader=({k: v.to(device) for k, v in batch.items()} for batch in dataloader),
        output_dir=args.output_dir,
        config=config,
        samples_per_shard=args.samples_per_shard,
        max_chunks=args.max_chunks,
    )
    print({"backbone": args.backbone, "tuples_written": total, "output_dir": args.output_dir})


if __name__ == "__main__":
    main()
