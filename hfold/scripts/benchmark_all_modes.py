"""Benchmark full / sliding / HFold on a real eval dataset.

Example:

    python -m hfold.scripts.benchmark_all_modes \
      --backbone pythia \
      --model-name EleutherAI/pythia-31m \
      --checkpoint-dir ./checkpoints/pythia_full_finetuning \
      --aux-dir ./checkpoints/aux \
      --dataset wikitext
"""
from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from hfold.config.schema import HFoldConfig, HFoldModelConfig, HFoldTrainingConfig
from hfold.integration.benchmark_runner import benchmark_three_modes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["pythia", "gpt2"], required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--checkpoint-dir", default=None, help="Local fine-tuned checkpoint directory.")
    parser.add_argument("--aux-dir", default=None, help="Directory with aux state_dicts.")
    parser.add_argument("--dataset", choices=["wikitext"], default="wikitext")
    parser.add_argument("--wikitext-config", default="wikitext-103-raw-v1")
    parser.add_argument("--cache-dir", default="./data")
    parser.add_argument("--chunk-len", type=int, default=512)
    parser.add_argument("--max-eval-batches", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-heap-size", type=int, default=16)
    parser.add_argument("--adapter-dim", type=int, default=256)
    parser.add_argument(
        "--embedding-model-type",
        choices=["autoencoder", "mean_identity", "mean_bottleneck"],
        default="autoencoder",
    )
    parser.add_argument(
        "--candidate-score-mode",
        choices=["attention", "hidden_dot"],
        default="hidden_dot",
    )
    parser.add_argument("--aux-fold-interval", type=int, default=1)
    parser.add_argument("--hfold-step-interval", type=int, default=1)
    parser.add_argument(
        "--hfold-eval-chunk-size",
        type=int,
        default=1,
        help="HFold eval only: number of next-token predictions per forward step.",
    )
    parser.add_argument(
        "--hfold-eval-use-kv-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="HFold eval only: toggle KV-cache path in benchmark runner.",
    )
    parser.add_argument("--allow-random-aux", action="store_true")
    return parser.parse_args()


def _resolve_device(device_str: str) -> str:
    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_str


def _aux_paths(aux_dir: str | None) -> dict[str, str | None]:
    if not aux_dir:
        return {"embedding": None, "relevancy": None, "adapters": None}
    return {
        "embedding": os.path.join(aux_dir, "embedding_autoencoder.pt"),
        "relevancy": os.path.join(aux_dir, "relevancy_transformer.pt"),
        "adapters": os.path.join(aux_dir, "adapters.pt"),
    }


def _build_dataloader(args: argparse.Namespace, tokenizer) -> DataLoader:
    from datasets import load_dataset

    if args.dataset == "wikitext":
        raw = load_dataset("wikitext", args.wikitext_config, cache_dir=args.cache_dir)
    else:
        raise ValueError(f"unsupported dataset: {args.dataset}")
    split = "validation" if "validation" in raw else "test" if "test" in raw else next(iter(raw.keys()))

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
        return {
            "input_ids": chunks,
            "attention_mask": [[1] * chunk_len for _ in chunks],
            "labels": chunks,
        }

    grouped = tokenized.map(group, batched=True, remove_columns=tokenized.column_names)
    rows = list(grouped.select(range(min(len(grouped), args.max_eval_batches))))

    def collate(batch_rows):
        ids = torch.tensor([row["input_ids"] for row in batch_rows], dtype=torch.long)
        mask = torch.tensor([row["attention_mask"] for row in batch_rows], dtype=torch.long)
        labels = torch.tensor([row["labels"] for row in batch_rows], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}

    return DataLoader(rows, batch_size=1, shuffle=False, collate_fn=collate)


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    aux = _aux_paths(args.aux_dir)
    if not args.allow_random_aux and any(aux[k] is None or not os.path.exists(aux[k]) for k in ("embedding", "relevancy", "adapters")):
        raise RuntimeError(
            "Aux checkpoints missing. Train them with `train_aux_models.py` and pass --aux-dir, "
            "or pass --allow-random-aux to explicitly use random weights for HFold."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataloader = _build_dataloader(args, tokenizer)

    # Hidden size and num_heads are auto-detected inside the runner from the
    # actual model config; the values here only seed adapter_dim and heap.
    config = HFoldConfig(
        model=HFoldModelConfig(
            hidden_size=1,  # placeholder; runner overrides from model.config
            num_heads=1,
            max_heap_size=args.max_heap_size,
            top_w=args.max_heap_size,
            pop_k=args.max_heap_size,
            adapter_dim=args.adapter_dim,
            embedding_model_type=args.embedding_model_type,
            candidate_score_mode=args.candidate_score_mode,
            aux_fold_interval=max(1, int(args.aux_fold_interval)),
            hfold_step_interval=max(1, int(args.hfold_step_interval)),
        ),
        training=HFoldTrainingConfig(),
    )

    results = benchmark_three_modes(
        backbone=args.backbone,
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_dir,
        dataloader=dataloader,
        config=config,
        device=device,
        hfold_eval_use_kv_cache=bool(args.hfold_eval_use_kv_cache),
        hfold_eval_chunk_size=max(1, int(args.hfold_eval_chunk_size)),
        embedding_checkpoint_path=aux["embedding"] if args.aux_dir else None,
        relevancy_checkpoint_path=aux["relevancy"] if args.aux_dir else None,
        adapters_checkpoint_path=aux["adapters"] if args.aux_dir else None,
    )
    for r in results:
        print(f"{r.mode:>16}  loss={r.loss:.4f}  ppl={r.perplexity:.4f}  tok/s={r.tokens_per_second:.2f}")


if __name__ == "__main__":
    main()
