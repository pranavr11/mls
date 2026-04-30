"""Modal end-to-end benchmark matrix for full/sliding/HFold.

Runs the following for each (backbone, dataset) pair:
1) Base (no fine-tune): benchmark full/sliding/hfold
2) Full-attention fine-tune: benchmark full/sliding/hfold
3) Sliding-window fine-tune: benchmark full/sliding/hfold

Default model/dataset matrix:
- pythia: EleutherAI/pythia-31m on wikitext + scrolls(gov_report)
- gpt2: gpt2 on wikitext + scrolls(gov_report)

Execution:
  modal run hfold/scripts/modal_full_matrix.py --help
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import modal


APP_NAME = "mlsfinal-hfold-matrix"
REPO_ROOT = Path(__file__).resolve().parents[2]
REMOTE_REPO_ROOT = Path("/root/project")
VOLUME_PATH = Path("/mnt/mlsfinal")

volume = modal.Volume.from_name("mlsfinal-hfold-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.36,<4.45",
        "datasets==2.18.0",
        "evaluate",
        "tqdm",
        "pandas",
        "python-dotenv",
        "absl-py",
        "nltk",
        "rouge-score",
    )
    .add_local_dir(str(REPO_ROOT), remote_path=str(REMOTE_REPO_ROOT))
    .workdir(str(REMOTE_REPO_ROOT))
)

app = modal.App(APP_NAME)


@dataclass
class RunRow:
    timestamp: int
    backbone: str
    model_name: str
    dataset: str
    dataset_config: str
    phase: str
    mode: str
    loss: float
    ppl: float
    tok_s: float
    checkpoint_path: str | None


def _set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device: str) -> str:
    import torch

    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dataset_label(dataset_name: str, dataset_config: str) -> str:
    return dataset_name if dataset_name == "wikitext" else f"scrolls_{dataset_config}"


def _load_raw_dataset(dataset_name: str, dataset_config: str, cache_dir: str):
    from datasets import load_dataset

    if dataset_name == "wikitext":
        return load_dataset("wikitext", dataset_config, cache_dir=cache_dir)
    if dataset_name == "scrolls":
        return load_dataset(
            "tau/scrolls",
            dataset_config,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _normalize_text_batch(examples: dict, dataset_name: str) -> list[str]:
    if dataset_name == "wikitext":
        return [t for t in examples["text"] if isinstance(t, str)]

    # SCROLLS-style row shape typically has "input" and optional "output".
    inputs = examples.get("input", [""] * len(next(iter(examples.values()))))
    outputs = examples.get("output", [""] * len(inputs))
    out_texts: list[str] = []
    for inp, out in zip(inputs, outputs):
        inp_s = inp if isinstance(inp, str) else ""
        out_s = out if isinstance(out, str) else ""
        out_texts.append(f"Document:\n{inp_s}\n\nSummary:\n{out_s}")
    return out_texts


def _group_texts(examples: dict, block_size: int) -> dict:
    concat_ids: list[int] = []
    for ids in examples["input_ids"]:
        concat_ids.extend(ids)
    n = (len(concat_ids) // block_size) * block_size
    chunks = [concat_ids[i : i + block_size] for i in range(0, n, block_size)]
    return {
        "input_ids": chunks,
        "attention_mask": [[1] * block_size for _ in chunks],
        "labels": chunks,
    }


def _build_lm_dataloaders(
    *,
    tokenizer,
    dataset_name: str,
    dataset_config: str,
    cache_dir: str,
    max_length: int,
    train_batch_size: int,
    eval_batch_size: int,
    max_train_batches: int,
    max_eval_batches: int,
):
    import torch
    from torch.utils.data import DataLoader

    raw = _load_raw_dataset(dataset_name, dataset_config, cache_dir)
    train_split = raw["train"]
    eval_key = "validation" if "validation" in raw else "test"
    eval_split = raw[eval_key]

    # Tokenize to contiguous LM chunks for both datasets.
    def tokenize(batch):
        texts = _normalize_text_batch(batch, dataset_name)
        return tokenizer(texts, add_special_tokens=False)

    tokenized_train = train_split.map(tokenize, batched=True, remove_columns=train_split.column_names)
    tokenized_eval = eval_split.map(tokenize, batched=True, remove_columns=eval_split.column_names)

    grouped_train = tokenized_train.map(
        lambda ex: _group_texts(ex, max_length),
        batched=True,
        remove_columns=tokenized_train.column_names,
    )
    grouped_eval = tokenized_eval.map(
        lambda ex: _group_texts(ex, max_length),
        batched=True,
        remove_columns=tokenized_eval.column_names,
    )

    train_rows = list(grouped_train.select(range(min(len(grouped_train), max_train_batches))))
    eval_rows = list(grouped_eval.select(range(min(len(grouped_eval), max_eval_batches))))

    def collate(batch_rows):
        ids = torch.tensor([row["input_ids"] for row in batch_rows], dtype=torch.long)
        mask = torch.tensor([row["attention_mask"] for row in batch_rows], dtype=torch.long)
        labels = torch.tensor([row["labels"] for row in batch_rows], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}

    train_loader = DataLoader(train_rows, batch_size=train_batch_size, shuffle=True, collate_fn=collate)
    eval_loader = DataLoader(eval_rows, batch_size=eval_batch_size, shuffle=False, collate_fn=collate)
    return train_loader, eval_loader


def _make_hfold_config(max_heap_size: int, adapter_dim: int, aux_fold_interval: int, hfold_step_interval: int):
    from hfold.config.schema import HFoldConfig, HFoldModelConfig, HFoldTrainingConfig

    return HFoldConfig(
        model=HFoldModelConfig(
            hidden_size=1,
            num_heads=1,
            max_heap_size=max_heap_size,
            top_w=max_heap_size,
            pop_k=max_heap_size,
            adapter_dim=adapter_dim,
            aux_fold_interval=max(1, int(aux_fold_interval)),
            hfold_step_interval=max(1, int(hfold_step_interval)),
        ),
        training=HFoldTrainingConfig(),
    )


def _apply_sliding_window_all_layers(model, window_size: int) -> None:
    from hfold.integration.benchmark_runner import _apply_sliding_window

    _apply_sliding_window(model, window_size)


def _unwrap_sliding_window_all_layers(model) -> None:
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        for layer in model.gpt_neox.layers:
            attn = layer.attention
            if hasattr(attn, "original_attention"):
                layer.attention = attn.original_attention
        return
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        for layer in model.transformer.h:
            attn = layer.attn
            if hasattr(attn, "original_attention"):
                layer.attn = attn.original_attention
        return
    raise ValueError("Unsupported model architecture for sliding-window unwrapping.")


def _train_one_epoch(model, dataloader, optimizer, scheduler, device: str, desc: str) -> float:
    import torch
    from tqdm.auto import tqdm

    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=desc, leave=False)
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += float(loss.item())
        pbar.set_postfix({"loss": f"{float(loss.item()):.4f}"})

    return total_loss / max(len(dataloader), 1)


@modal.function(
    image=image,
    gpu="B200",
    timeout=60 * 60 * 24,
    volumes={str(VOLUME_PATH): volume},
)
def run_experiments(
    *,
    model_csv: str,
    dataset_csv: str,
    scrolls_config_csv: str,
    seed: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    warmup_ratio: float,
    train_batch_size: int,
    eval_batch_size: int,
    max_length: int,
    max_train_batches: int,
    max_eval_batches: int,
    sliding_window_size: int,
    max_heap_size: int,
    adapter_dim: int,
    aux_fold_interval: int,
    hfold_step_interval: int,
    hfold_eval_use_kv_cache: bool,
    hfold_eval_chunk_size: int,
    output_subdir: str,
) -> dict:
    import torch
    from tqdm.auto import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW, get_cosine_schedule_with_warmup

    from hfold.integration.benchmark_runner import benchmark_three_modes

    os.chdir(str(REMOTE_REPO_ROOT))
    _set_seed(seed)
    device = _resolve_device("auto")

    models: list[tuple[str, str]] = []
    for raw in model_csv.split(","):
        name = raw.strip().lower()
        if not name:
            continue
        if name == "pythia":
            models.append(("pythia", "EleutherAI/pythia-31m"))
        elif name == "gpt2":
            models.append(("gpt2", "gpt2"))
        else:
            raise ValueError(f"Unsupported model key: {raw}")

    datasets = [d.strip().lower() for d in dataset_csv.split(",") if d.strip()]
    scroll_cfgs = [c.strip() for c in scrolls_config_csv.split(",") if c.strip()]
    if not datasets:
        raise ValueError("No datasets selected")
    if "scrolls" in datasets and not scroll_cfgs:
        raise ValueError("scrolls selected but --scrolls-configs is empty")

    timestamp = int(time.time())
    out_root = VOLUME_PATH / output_subdir
    ckpt_root = out_root / "checkpoints"
    results_root = out_root / "results"
    cache_root = out_root / "hf_cache"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    # Build explicit experiment grid.
    combos: list[tuple[str, str, str, str]] = []
    for backbone, model_name in models:
        for ds in datasets:
            if ds == "wikitext":
                combos.append((backbone, model_name, "wikitext", "wikitext-103-raw-v1"))
            elif ds == "scrolls":
                for cfg in scroll_cfgs:
                    combos.append((backbone, model_name, "scrolls", cfg))
            else:
                raise ValueError(f"Unsupported dataset key: {ds}")

    all_rows: list[RunRow] = []
    total_steps = len(combos) * 3  # base + full_ft + sliding_ft per combo
    outer = tqdm(total=total_steps, desc="Experiment matrix", position=0)

    for backbone, model_name, dataset_name, dataset_config in combos:
        combo_label = f"{backbone}:{dataset_name}:{dataset_config}"
        print(f"\n=== {combo_label} ===")

        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_root))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        train_loader, eval_loader = _build_lm_dataloaders(
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            cache_dir=str(cache_root),
            max_length=max_length,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            max_train_batches=max_train_batches,
            max_eval_batches=max_eval_batches,
        )

        # 1) Base model benchmark
        cfg_base = _make_hfold_config(
            max_heap_size=max_heap_size,
            adapter_dim=adapter_dim,
            aux_fold_interval=aux_fold_interval,
            hfold_step_interval=hfold_step_interval,
        )
        base_results = benchmark_three_modes(
            backbone=backbone,
            model_name=model_name,
            checkpoint_path=None,
            dataloader=eval_loader,
            config=cfg_base,
            device=device,
            sliding_window_size=sliding_window_size,
            hfold_eval_use_kv_cache=hfold_eval_use_kv_cache,
            hfold_eval_chunk_size=hfold_eval_chunk_size,
        )
        for r in base_results:
            all_rows.append(
                RunRow(
                    timestamp=timestamp,
                    backbone=backbone,
                    model_name=model_name,
                    dataset=dataset_name,
                    dataset_config=dataset_config,
                    phase="base",
                    mode=r.mode,
                    loss=float(r.loss),
                    ppl=float(r.perplexity),
                    tok_s=float(r.tokens_per_second),
                    checkpoint_path=None,
                )
            )
            print(f"  [base] {r.mode:>14}  loss={r.loss:.4f}  ppl={r.perplexity:.4f}  tok/s={r.tokens_per_second:.2f}")
        outer.update(1)

        # 2) Full-attention fine-tune
        full_tag = f"{backbone}_{_dataset_label(dataset_name, dataset_config)}_full_ep{epochs}"
        full_ckpt = ckpt_root / full_tag
        model_full = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=str(cache_root)).to(device)
        if tokenizer.pad_token_id is not None:
            model_full.config.pad_token_id = tokenizer.pad_token_id

        total_steps_train = len(train_loader) * max(1, epochs)
        optimizer = AdamW(model_full.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, int(total_steps_train * warmup_ratio)),
            num_training_steps=max(1, total_steps_train),
        )
        epoch_bar = tqdm(range(epochs), desc=f"FT(full) {combo_label}", position=1, leave=False)
        for epoch_idx in epoch_bar:
            train_loss = _train_one_epoch(
                model_full,
                train_loader,
                optimizer,
                scheduler,
                device,
                desc=f"train full e{epoch_idx + 1}/{epochs}",
            )
            epoch_bar.set_postfix({"train_loss": f"{train_loss:.4f}"})

        full_ckpt.mkdir(parents=True, exist_ok=True)
        model_full.save_pretrained(str(full_ckpt))
        tokenizer.save_pretrained(str(full_ckpt))

        cfg_full = _make_hfold_config(
            max_heap_size=max_heap_size,
            adapter_dim=adapter_dim,
            aux_fold_interval=aux_fold_interval,
            hfold_step_interval=hfold_step_interval,
        )
        full_ft_results = benchmark_three_modes(
            backbone=backbone,
            model_name=model_name,
            checkpoint_path=str(full_ckpt),
            dataloader=eval_loader,
            config=cfg_full,
            device=device,
            sliding_window_size=sliding_window_size,
            hfold_eval_use_kv_cache=hfold_eval_use_kv_cache,
            hfold_eval_chunk_size=hfold_eval_chunk_size,
        )
        for r in full_ft_results:
            all_rows.append(
                RunRow(
                    timestamp=timestamp,
                    backbone=backbone,
                    model_name=model_name,
                    dataset=dataset_name,
                    dataset_config=dataset_config,
                    phase="fine_tuned_full",
                    mode=r.mode,
                    loss=float(r.loss),
                    ppl=float(r.perplexity),
                    tok_s=float(r.tokens_per_second),
                    checkpoint_path=str(full_ckpt),
                )
            )
            print(
                f"  [ft-full] {r.mode:>11}  loss={r.loss:.4f}  "
                f"ppl={r.perplexity:.4f}  tok/s={r.tokens_per_second:.2f}"
            )
        outer.update(1)

        # 3) Sliding-window fine-tune
        sliding_tag = f"{backbone}_{_dataset_label(dataset_name, dataset_config)}_sliding_ep{epochs}"
        sliding_ckpt = ckpt_root / sliding_tag
        model_sliding = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=str(cache_root)).to(device)
        if tokenizer.pad_token_id is not None:
            model_sliding.config.pad_token_id = tokenizer.pad_token_id
        _apply_sliding_window_all_layers(model_sliding, sliding_window_size)

        optimizer_sw = AdamW(model_sliding.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler_sw = get_cosine_schedule_with_warmup(
            optimizer_sw,
            num_warmup_steps=max(1, int(total_steps_train * warmup_ratio)),
            num_training_steps=max(1, total_steps_train),
        )
        epoch_bar_sw = tqdm(range(epochs), desc=f"FT(sliding) {combo_label}", position=1, leave=False)
        for epoch_idx in epoch_bar_sw:
            train_loss = _train_one_epoch(
                model_sliding,
                train_loader,
                optimizer_sw,
                scheduler_sw,
                device,
                desc=f"train sliding e{epoch_idx + 1}/{epochs}",
            )
            epoch_bar_sw.set_postfix({"train_loss": f"{train_loss:.4f}"})

        # Remove wrappers before save so checkpoint key names match base architecture.
        _unwrap_sliding_window_all_layers(model_sliding)

        sliding_ckpt.mkdir(parents=True, exist_ok=True)
        model_sliding.save_pretrained(str(sliding_ckpt))
        tokenizer.save_pretrained(str(sliding_ckpt))

        cfg_sliding = _make_hfold_config(
            max_heap_size=max_heap_size,
            adapter_dim=adapter_dim,
            aux_fold_interval=aux_fold_interval,
            hfold_step_interval=hfold_step_interval,
        )
        sliding_ft_results = benchmark_three_modes(
            backbone=backbone,
            model_name=model_name,
            checkpoint_path=str(sliding_ckpt),
            dataloader=eval_loader,
            config=cfg_sliding,
            device=device,
            sliding_window_size=sliding_window_size,
            hfold_eval_use_kv_cache=hfold_eval_use_kv_cache,
            hfold_eval_chunk_size=hfold_eval_chunk_size,
        )
        for r in sliding_ft_results:
            all_rows.append(
                RunRow(
                    timestamp=timestamp,
                    backbone=backbone,
                    model_name=model_name,
                    dataset=dataset_name,
                    dataset_config=dataset_config,
                    phase="fine_tuned_sliding",
                    mode=r.mode,
                    loss=float(r.loss),
                    ppl=float(r.perplexity),
                    tok_s=float(r.tokens_per_second),
                    checkpoint_path=str(sliding_ckpt),
                )
            )
            print(
                f"  [ft-sliding] {r.mode:>8}  loss={r.loss:.4f}  "
                f"ppl={r.perplexity:.4f}  tok/s={r.tokens_per_second:.2f}"
            )
        outer.update(1)

        # Explicitly free GPU memory between combos.
        del model_full, model_sliding
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    outer.close()

    rows_json = [asdict(r) for r in all_rows]
    json_path = results_root / f"matrix_results_{timestamp}.json"
    jsonl_path = results_root / f"matrix_results_{timestamp}.jsonl"
    csv_path = results_root / f"matrix_results_{timestamp}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows_json, f, indent=2, sort_keys=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows_json:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    fieldnames = list(rows_json[0].keys()) if rows_json else [
        "timestamp",
        "backbone",
        "model_name",
        "dataset",
        "dataset_config",
        "phase",
        "mode",
        "loss",
        "ppl",
        "tok_s",
        "checkpoint_path",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_json)

    summary: dict[str, dict] = {}
    for row in rows_json:
        key = f"{row['backbone']}|{row['dataset']}|{row['dataset_config']}|{row['phase']}"
        summary.setdefault(key, {})[row["mode"]] = {
            "ppl": row["ppl"],
            "tok_s": row["tok_s"],
        }

    volume.commit()
    return {
        "device": device,
        "rows": len(rows_json),
        "output_root": str(out_root),
        "json": str(json_path),
        "jsonl": str(jsonl_path),
        "csv": str(csv_path),
        "summary": summary,
    }


@app.local_entrypoint()
def main(*arglist: str) -> None:
    parser = argparse.ArgumentParser(description="Run full/sliding/HFold matrix on Modal B200.")
    parser.add_argument("--models", default="pythia,gpt2", help="CSV from: pythia,gpt2")
    parser.add_argument("--datasets", default="wikitext,scrolls", help="CSV from: wikitext,scrolls")
    parser.add_argument(
        "--scrolls-configs",
        default="gov_report",
        help="CSV SCROLLS configs (e.g. gov_report,summ_screen_fd,qmsum)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-train-batches", type=int, default=256)
    parser.add_argument("--max-eval-batches", type=int, default=32)
    parser.add_argument("--sliding-window-size", type=int, default=256)
    parser.add_argument("--max-heap-size", type=int, default=16)
    parser.add_argument("--adapter-dim", type=int, default=256)
    parser.add_argument("--aux-fold-interval", type=int, default=4)
    parser.add_argument("--hfold-step-interval", type=int, default=1)
    parser.add_argument(
        "--hfold-eval-use-kv-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use HFold KV-cache eval path; default False for cleaner speed/accuracy debugging.",
    )
    parser.add_argument("--hfold-eval-chunk-size", type=int, default=8)
    parser.add_argument("--output-subdir", default="modal_matrix")

    args = parser.parse_args(arglist)

    result = run_experiments.remote(
        model_csv=args.models,
        dataset_csv=args.datasets,
        scrolls_config_csv=args.scrolls_configs,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        max_length=args.max_length,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        sliding_window_size=args.sliding_window_size,
        max_heap_size=args.max_heap_size,
        adapter_dim=args.adapter_dim,
        aux_fold_interval=args.aux_fold_interval,
        hfold_step_interval=args.hfold_step_interval,
        hfold_eval_use_kv_cache=bool(args.hfold_eval_use_kv_cache),
        hfold_eval_chunk_size=args.hfold_eval_chunk_size,
        output_subdir=args.output_subdir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
