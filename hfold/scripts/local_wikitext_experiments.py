"""Local end-to-end HFold experiment runner on a small WikiText slice.

This script is designed to replace ad-hoc notebook runs with a repeatable loop:
1) (Optional) prepare aux checkpoints by extracting hidden states + training aux models
2) run baseline benchmark (full / sliding / hfold) per checkpoint
3) sweep HFold-only knobs (`aux_fold_interval`, `hfold_eval_use_kv_cache`) repeatedly
4) append every run to a JSONL log for quick comparison
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Iterable

import torch
from torch.utils.data import DataLoader

from hfold.config.schema import HFoldConfig, HFoldModelConfig, HFoldTrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["pythia", "gpt2"], default="pythia")
    parser.add_argument("--model-name", default="EleutherAI/pythia-14m")
    parser.add_argument("--cache-dir", default="./data")
    parser.add_argument("--checkpoints-dir", default="./checkpoints")
    parser.add_argument("--full-checkpoint", default=None)
    parser.add_argument("--sliding-checkpoint", default=None)
    parser.add_argument("--skip-full-checkpoint", action="store_true")
    parser.add_argument("--skip-sliding-checkpoint", action="store_true")
    parser.add_argument("--device", default="auto")

    parser.add_argument("--chunk-len", type=int, default=512)
    parser.add_argument("--max-eval-batches", type=int, default=8)
    parser.add_argument("--sliding-window-size", type=int, default=64)

    parser.add_argument("--max-heap-size", type=int, default=16)
    parser.add_argument("--adapter-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)

    parser.add_argument("--aux-dir", default="./checkpoints/aux")
    parser.add_argument("--aux-extract-dir", default="./checkpoints/aux_extracted/pythia")
    parser.add_argument("--no-prepare-aux", action="store_true")
    parser.add_argument("--force-retrain-aux", action="store_true")
    parser.add_argument("--allow-random-aux", action="store_true")

    parser.add_argument("--extract-max-chunks", type=int, default=64)
    parser.add_argument(
        "--extract-max-raw-rows",
        type=int,
        default=50000,
        help="Cap WikiText train rows scanned for aux extraction (faster local loops).",
    )
    parser.add_argument("--extract-samples-per-shard", type=int, default=128)
    parser.add_argument("--extract-num-anchors", type=int, default=4)
    parser.add_argument("--extract-seed", type=int, default=42)

    parser.add_argument("--aux-epochs", type=int, default=2)
    parser.add_argument("--aux-batch-size", type=int, default=16)
    parser.add_argument("--aux-lr", type=float, default=3e-4)
    parser.add_argument("--aux-max-steps", type=int, default=400)

    parser.add_argument("--hfold-kv-cache-modes", default="true,false")
    parser.add_argument("--aux-fold-intervals", default="1,2,4,8")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--results-jsonl", default="./checkpoints/local_wikitext_experiments.jsonl")
    return parser.parse_args()


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def _parse_bool_csv(csv_value: str) -> list[bool]:
    out: list[bool] = []
    for tok in csv_value.split(","):
        v = tok.strip().lower()
        if not v:
            continue
        if v in {"1", "true", "t", "yes", "y"}:
            out.append(True)
            continue
        if v in {"0", "false", "f", "no", "n"}:
            out.append(False)
            continue
        raise ValueError(f"invalid boolean token in CSV: {tok!r}")
    if not out:
        raise ValueError("empty boolean CSV value")
    return out


def _parse_int_csv(csv_value: str) -> list[int]:
    out = [int(tok.strip()) for tok in csv_value.split(",") if tok.strip()]
    if not out:
        raise ValueError("empty integer CSV value")
    return out


def _infer_hidden_size(model_config) -> int:
    for key in ("hidden_size", "n_embd", "d_model", "dim"):
        value = getattr(model_config, key, None)
        if value is not None:
            return int(value)
    raise ValueError("could not infer hidden size from model config")


def _load_model(
    *,
    backbone: str,
    model_name: str,
    checkpoint_path: str | None,
    cache_dir: str,
    device: torch.device,
) -> torch.nn.Module:
    from transformers import AutoModelForCausalLM
    from hfold.integration.checkpoint_utils import load_gpt_neox_causal_lm_from_folder

    if checkpoint_path and backbone == "pythia":
        model = load_gpt_neox_causal_lm_from_folder(checkpoint_path, cache_dir=cache_dir)
    elif checkpoint_path:
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, attn_implementation="eager")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, attn_implementation="eager")
    return model.to(device)


def _build_eval_loader(args: argparse.Namespace, tokenizer) -> DataLoader:
    from datasets import load_dataset

    raw = load_dataset("wikitext", "wikitext-103-raw-v1", cache_dir=args.cache_dir)
    split = "validation" if "validation" in raw else "test" if "test" in raw else next(iter(raw.keys()))

    def tokenize(batch):
        return tokenizer(batch["text"], add_special_tokens=False)

    tokenized = raw[split].map(tokenize, batched=True, remove_columns=raw[split].column_names)

    def group(batch):
        concat: list[int] = []
        for ids in batch["input_ids"]:
            concat.extend(ids)
        n = (len(concat) // args.chunk_len) * args.chunk_len
        chunks = [concat[i : i + args.chunk_len] for i in range(0, n, args.chunk_len)]
        return {
            "input_ids": chunks,
            "attention_mask": [[1] * args.chunk_len for _ in chunks],
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


def _build_extract_loader(args: argparse.Namespace, tokenizer) -> DataLoader:
    from datasets import load_dataset

    raw = load_dataset("wikitext", "wikitext-103-raw-v1", cache_dir=args.cache_dir)
    split = "train" if "train" in raw else next(iter(raw.keys()))
    split_ds = raw[split]
    if int(args.extract_max_raw_rows) > 0:
        take = min(len(split_ds), int(args.extract_max_raw_rows))
        split_ds = split_ds.select(range(take))

    def tokenize(batch):
        return tokenizer(batch["text"], add_special_tokens=False)

    tokenized = split_ds.map(tokenize, batched=True, remove_columns=split_ds.column_names)

    def group(batch):
        concat: list[int] = []
        for ids in batch["input_ids"]:
            concat.extend(ids)
        n = (len(concat) // args.chunk_len) * args.chunk_len
        chunks = [concat[i : i + args.chunk_len] for i in range(0, n, args.chunk_len)]
        return {
            "input_ids": chunks,
            "attention_mask": [[1] * args.chunk_len for _ in chunks],
        }

    grouped = tokenized.map(group, batched=True, remove_columns=tokenized.column_names)

    def collate(rows):
        ids = torch.tensor([row["input_ids"] for row in rows], dtype=torch.long)
        mask = torch.tensor([row["attention_mask"] for row in rows], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask}

    return DataLoader(grouped, batch_size=1, shuffle=False, collate_fn=collate)


def _aux_paths(aux_dir: str) -> dict[str, str]:
    return {
        "embedding": os.path.join(aux_dir, "embedding_autoencoder.pt"),
        "relevancy": os.path.join(aux_dir, "relevancy_transformer.pt"),
        "adapters": os.path.join(aux_dir, "adapters.pt"),
    }


def _aux_ready(paths: dict[str, str]) -> bool:
    return all(os.path.exists(p) for p in paths.values())


def _prepare_aux_if_needed(
    *,
    args: argparse.Namespace,
    device: torch.device,
    backbone_checkpoint_for_extract: str | None,
) -> dict[str, str] | None:
    from transformers import AutoTokenizer
    from hfold.data.extract_hidden_states import ExtractionConfig, extract_to_shards
    from hfold.data.hidden_state_dataset import HiddenStateShardDataset
    from hfold.training.train_embedding import train_embedding_model
    from hfold.training.train_relevancy import train_relevancy_model

    if args.allow_random_aux:
        return None
    paths = _aux_paths(args.aux_dir)
    if args.no_prepare_aux and not _aux_ready(paths):
        raise RuntimeError(
            "aux checkpoints missing and --no-prepare-aux was set; "
            "remove that flag or pass --allow-random-aux"
        )
    if _aux_ready(paths) and not args.force_retrain_aux:
        print({"aux_status": "reusing_existing", "aux_dir": args.aux_dir})
        return paths
    if args.no_prepare_aux:
        return paths

    os.makedirs(args.aux_dir, exist_ok=True)
    os.makedirs(args.aux_extract_dir, exist_ok=True)
    print({"aux_status": "training", "aux_dir": args.aux_dir, "extract_dir": args.aux_extract_dir})

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _load_model(
        backbone=args.backbone,
        model_name=args.model_name,
        checkpoint_path=backbone_checkpoint_for_extract,
        cache_dir=args.cache_dir,
        device=device,
    )
    hidden_size = _infer_hidden_size(model.config)

    extract_loader = _build_extract_loader(args, tokenizer)
    extract_cfg = ExtractionConfig(
        backbone=args.backbone,
        chunk_len=args.chunk_len,
        max_heap_size=args.max_heap_size,
        num_anchors_per_chunk=args.extract_num_anchors,
        seed=args.extract_seed,
    )
    written = extract_to_shards(
        model=model,
        dataloader=({k: v.to(device) for k, v in batch.items()} for batch in extract_loader),
        output_dir=args.aux_extract_dir,
        config=extract_cfg,
        samples_per_shard=args.extract_samples_per_shard,
        max_chunks=args.extract_max_chunks,
    )
    print({"aux_samples_written": int(written), "extract_dir": args.aux_extract_dir})
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    aux_cfg = HFoldConfig(
        model=HFoldModelConfig(
            hidden_size=hidden_size,
            num_heads=args.num_heads,
            max_heap_size=args.max_heap_size,
            top_w=args.max_heap_size,
            pop_k=args.max_heap_size,
            adapter_dim=args.adapter_dim,
        ),
        training=HFoldTrainingConfig(
            learning_rate=args.aux_lr,
            num_epochs=args.aux_epochs,
            batch_size=args.aux_batch_size,
            max_steps=args.aux_max_steps,
            seed=args.extract_seed,
            device=str(device),
        ),
    )
    dataset = HiddenStateShardDataset([args.aux_extract_dir])
    print({"aux_num_shards_total": len(dataset)})
    emb_artifacts = train_embedding_model(
        config=aux_cfg,
        dataset=dataset,
        backbone_dims={args.backbone: hidden_size},
    )
    rel_artifacts = train_relevancy_model(
        config=aux_cfg,
        dataset=dataset,
        embedding_model=emb_artifacts.model,
        adapters=emb_artifacts.adapters,
    )

    torch.save(emb_artifacts.model.state_dict(), paths["embedding"])
    torch.save(emb_artifacts.adapters.state_dict(), paths["adapters"])
    torch.save(rel_artifacts.model.state_dict(), paths["relevancy"])
    print(
        {
            "aux_status": "saved",
            "embedding_loss": float(emb_artifacts.final_loss),
            "relevancy_loss": float(rel_artifacts.final_loss),
            "aux_dir": args.aux_dir,
        }
    )
    return paths


def _write_jsonl(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _default_checkpoint_paths(args: argparse.Namespace) -> tuple[str, str]:
    model_tag = args.model_name.replace("/", "--")
    full_ckpt = os.path.join(args.checkpoints_dir, f"last3_full_attention_{model_tag}")
    sliding_ckpt = os.path.join(args.checkpoints_dir, f"last3_full_attention_{model_tag}_sw")
    return full_ckpt, sliding_ckpt


def _build_benchmark_config(args: argparse.Namespace, aux_fold_interval: int) -> HFoldConfig:
    return HFoldConfig(
        model=HFoldModelConfig(
            hidden_size=1,  # runner overwrites from model config
            num_heads=1,  # runner overwrites from model config
            max_heap_size=args.max_heap_size,
            top_w=args.max_heap_size,
            pop_k=args.max_heap_size,
            adapter_dim=args.adapter_dim,
            aux_fold_interval=max(1, int(aux_fold_interval)),
        ),
        training=HFoldTrainingConfig(),
    )


def _row_base(args: argparse.Namespace) -> dict:
    return {
        "ts": int(time.time()),
        "model_name": args.model_name,
        "backbone": args.backbone,
        "chunk_len": args.chunk_len,
        "max_eval_batches": args.max_eval_batches,
        "sliding_window_size": args.sliding_window_size,
        "max_heap_size": args.max_heap_size,
        "adapter_dim": args.adapter_dim,
    }


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer
    from hfold.integration.benchmark_runner import benchmark_three_modes, eval_hfold_only

    device = _resolve_device(args.device)
    kv_modes = _parse_bool_csv(args.hfold_kv_cache_modes)
    aux_intervals = _parse_int_csv(args.aux_fold_intervals)
    if args.repeats <= 0:
        raise ValueError("--repeats must be >= 1")

    full_ckpt_default, sliding_ckpt_default = _default_checkpoint_paths(args)
    ckpt_full = args.full_checkpoint or full_ckpt_default
    ckpt_sliding = args.sliding_checkpoint or sliding_ckpt_default

    print(
        {
            "device": str(device),
            "full_checkpoint": ckpt_full,
            "sliding_checkpoint": ckpt_sliding,
            "results_jsonl": args.results_jsonl,
        }
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eval_loader = _build_eval_loader(args, tokenizer)

    aux_paths = _prepare_aux_if_needed(
        args=args,
        device=device,
        backbone_checkpoint_for_extract=ckpt_full if os.path.isdir(ckpt_full) else None,
    )
    emb = None if aux_paths is None else aux_paths["embedding"]
    rel = None if aux_paths is None else aux_paths["relevancy"]
    adp = None if aux_paths is None else aux_paths["adapters"]

    checkpoints: list[tuple[str, str]] = []
    if not args.skip_full_checkpoint and os.path.isdir(ckpt_full):
        checkpoints.append(("full-attention fine-tuned", ckpt_full))
    if not args.skip_sliding_checkpoint and os.path.isdir(ckpt_sliding):
        checkpoints.append(("sliding-window fine-tuned", ckpt_sliding))
    if not checkpoints:
        raise RuntimeError("no checkpoint paths found; pass --full-checkpoint/--sliding-checkpoint")

    all_rows: list[dict] = []
    for label, ckpt in checkpoints:
        print(f"\n=== {label} ===\n  checkpoint: {ckpt}")
        for rep in range(args.repeats):
            baseline_cfg = _build_benchmark_config(args, aux_intervals[0])
            baseline_results = benchmark_three_modes(
                backbone=args.backbone,
                model_name=args.model_name,
                checkpoint_path=ckpt,
                dataloader=eval_loader,
                config=baseline_cfg,
                device=str(device),
                sliding_window_size=args.sliding_window_size,
                hfold_eval_use_kv_cache=kv_modes[0],
                embedding_checkpoint_path=emb,
                relevancy_checkpoint_path=rel,
                adapters_checkpoint_path=adp,
            )
            for res in baseline_results:
                print(
                    f"  rep={rep + 1:02d} {res.mode:>16}  loss={res.loss:.4f}  "
                    f"ppl={res.perplexity:.4f}  tok/s={res.tokens_per_second:.2f}"
                )
                row = _row_base(args)
                row.update(
                    {
                        "checkpoint_label": label,
                        "checkpoint_path": ckpt,
                        "repeat_idx": rep,
                        "mode": res.mode,
                        "loss": float(res.loss),
                        "ppl": float(res.perplexity),
                        "tok_s": float(res.tokens_per_second),
                        "hfold_eval_use_kv_cache": kv_modes[0] if res.mode == "hfold" else None,
                        "aux_fold_interval": aux_intervals[0] if res.mode == "hfold" else None,
                    }
                )
                all_rows.append(row)

            for interval in aux_intervals:
                for use_cache in kv_modes:
                    if interval == aux_intervals[0] and use_cache == kv_modes[0]:
                        continue
                    hcfg = _build_benchmark_config(args, interval)
                    hres = eval_hfold_only(
                        backbone=args.backbone,
                        model_name=args.model_name,
                        checkpoint_path=ckpt,
                        dataloader=eval_loader,
                        config=hcfg,
                        device=str(device),
                        sliding_window_size=args.sliding_window_size,
                        hfold_eval_use_kv_cache=use_cache,
                        embedding_checkpoint_path=emb,
                        relevancy_checkpoint_path=rel,
                        adapters_checkpoint_path=adp,
                        mode_label=f"hfold_aux{interval}_cache{int(use_cache)}",
                    )
                    print(
                        f"  rep={rep + 1:02d} {hres.mode:>16}  loss={hres.loss:.4f}  "
                        f"ppl={hres.perplexity:.4f}  tok/s={hres.tokens_per_second:.2f}"
                    )
                    row = _row_base(args)
                    row.update(
                        {
                            "checkpoint_label": label,
                            "checkpoint_path": ckpt,
                            "repeat_idx": rep,
                            "mode": "hfold",
                            "mode_label": hres.mode,
                            "loss": float(hres.loss),
                            "ppl": float(hres.perplexity),
                            "tok_s": float(hres.tokens_per_second),
                            "hfold_eval_use_kv_cache": bool(use_cache),
                            "aux_fold_interval": int(interval),
                        }
                    )
                    all_rows.append(row)

    _write_jsonl(args.results_jsonl, all_rows)
    print({"rows_written": len(all_rows), "results_jsonl": args.results_jsonl})

    # Small terminal summary for quick triage.
    best_speed = max((r for r in all_rows if r["mode"] == "hfold"), key=lambda r: r["tok_s"], default=None)
    best_ppl = min((r for r in all_rows if r["mode"] == "hfold"), key=lambda r: r["ppl"], default=None)
    if best_speed:
        print(
            {
                "best_hfold_speed": {
                    "mode_label": best_speed.get("mode_label", "hfold"),
                    "tok_s": best_speed["tok_s"],
                    "ppl": best_speed["ppl"],
                    "aux_fold_interval": best_speed.get("aux_fold_interval"),
                    "hfold_eval_use_kv_cache": best_speed.get("hfold_eval_use_kv_cache"),
                }
            }
        )
    if best_ppl:
        print(
            {
                "best_hfold_ppl": {
                    "mode_label": best_ppl.get("mode_label", "hfold"),
                    "tok_s": best_ppl["tok_s"],
                    "ppl": best_ppl["ppl"],
                    "aux_fold_interval": best_ppl.get("aux_fold_interval"),
                    "hfold_eval_use_kv_cache": best_ppl.get("hfold_eval_use_kv_cache"),
                }
            }
        )


if __name__ == "__main__":
    main()
