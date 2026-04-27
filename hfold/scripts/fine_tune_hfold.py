"""Fine-tune backbone + HFold stack with autoregressive runtime unrolling.

This script is meant for HFold-aware fine-tuning, unlike standard LM training
that calls the model once per sequence (which does not exercise HFold runtime
timesteps). We unroll next-token prediction step-by-step so heap updates happen.

Note: "differentiable_heap" here means gradients flow through selected heap
paths. Selection still uses hard top-k, so the objective is piecewise smooth.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from hfold.config.schema import HFoldConfig, HFoldModelConfig, HFoldTrainingConfig
from hfold.integration.gpt2_runner import build_gpt2_with_hfold
from hfold.integration.pythia_runner import build_pythia_with_hfold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["pythia", "gpt2"], default="pythia")
    parser.add_argument("--model-name", default="EleutherAI/pythia-14m")
    parser.add_argument("--checkpoint-dir", default=None, help="Backbone checkpoint dir to start from.")
    parser.add_argument("--cache-dir", default="./data")
    parser.add_argument("--output-dir", default="./checkpoints/hfold_finetune")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")

    parser.add_argument("--dataset", choices=["wikitext"], default="wikitext")
    parser.add_argument("--wikitext-config", default="wikitext-103-raw-v1")
    parser.add_argument("--chunk-len", type=int, default=256)
    parser.add_argument(
        "--max-train-raw-rows",
        type=int,
        default=50000,
        help="Cap raw train rows loaded before tokenization/grouping for faster iteration.",
    )
    parser.add_argument(
        "--max-eval-raw-rows",
        type=int,
        default=8000,
        help="Cap raw eval rows loaded before tokenization/grouping for faster iteration.",
    )
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--max-train-batches", type=int, default=16)
    parser.add_argument("--max-eval-batches", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--max-heap-size", type=int, default=16)
    parser.add_argument("--adapter-dim", type=int, default=256)
    parser.add_argument("--aux-fold-interval", type=int, default=1)
    parser.add_argument(
        "--embedding-model-type",
        choices=["autoencoder", "mean_identity", "mean_bottleneck"],
        default="autoencoder",
    )
    parser.add_argument(
        "--differentiable-heap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep heap tensors graph-connected across timesteps.",
    )

    parser.add_argument("--sliding-window-size", type=int, default=64)
    parser.add_argument(
        "--max-unroll-steps",
        type=int,
        default=64,
        help="Per sequence, train on at most this many next-token steps (for memory).",
    )

    parser.add_argument("--embedding-checkpoint", default=None)
    parser.add_argument("--relevancy-checkpoint", default=None)
    parser.add_argument("--adapters-checkpoint", default=None)
    parser.add_argument(
        "--train-scope",
        choices=["all", "backbone_only", "aux_only", "last_k_backbone_plus_aux"],
        default="all",
    )
    parser.add_argument("--last-k-layers", type=int, default=2)
    return parser.parse_args()


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_lm_dataloaders(args: argparse.Namespace, tokenizer) -> tuple[DataLoader, DataLoader]:
    from datasets import load_dataset

    if args.dataset != "wikitext":
        raise ValueError(f"unsupported dataset: {args.dataset}")
    raw = load_dataset("wikitext", args.wikitext_config, cache_dir=args.cache_dir)
    train_split = raw["train"]
    eval_key = "validation" if "validation" in raw else "test"
    eval_split = raw[eval_key]
    if int(args.max_train_raw_rows) > 0:
        train_split = train_split.select(range(min(len(train_split), int(args.max_train_raw_rows))))
    if int(args.max_eval_raw_rows) > 0:
        eval_split = eval_split.select(range(min(len(eval_split), int(args.max_eval_raw_rows))))

    def tokenize(batch):
        return tokenizer(batch["text"], add_special_tokens=False)

    tokenized_train = train_split.map(tokenize, batched=True, remove_columns=train_split.column_names)
    tokenized_eval = eval_split.map(tokenize, batched=True, remove_columns=eval_split.column_names)
    chunk_len = int(args.chunk_len)

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

    grouped_train = tokenized_train.map(group, batched=True, remove_columns=tokenized_train.column_names)
    grouped_eval = tokenized_eval.map(group, batched=True, remove_columns=tokenized_eval.column_names)
    train_rows = list(grouped_train.select(range(min(len(grouped_train), int(args.max_train_batches)))))
    eval_rows = list(grouped_eval.select(range(min(len(grouped_eval), int(args.max_eval_batches)))))

    def collate(rows):
        ids = torch.tensor([row["input_ids"] for row in rows], dtype=torch.long)
        mask = torch.tensor([row["attention_mask"] for row in rows], dtype=torch.long)
        labels = torch.tensor([row["labels"] for row in rows], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}

    train_loader = DataLoader(train_rows, batch_size=int(args.train_batch_size), shuffle=True, collate_fn=collate)
    eval_loader = DataLoader(eval_rows, batch_size=int(args.eval_batch_size), shuffle=False, collate_fn=collate)
    return train_loader, eval_loader


def _build_hfold_bundle(args: argparse.Namespace, config: HFoldConfig):
    common = dict(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_dir,
        config=config,
        cache_dir=args.cache_dir,
        embedding_checkpoint_path=args.embedding_checkpoint,
        relevancy_checkpoint_path=args.relevancy_checkpoint,
        adapters_checkpoint_path=args.adapters_checkpoint,
    )
    if args.backbone == "pythia":
        return build_pythia_with_hfold(**common)
    if args.backbone == "gpt2":
        return build_gpt2_with_hfold(**common)
    raise ValueError(f"unsupported backbone: {args.backbone}")


def _set_trainable_scope(model: torch.nn.Module, scope: str, last_k_layers: int) -> None:
    for p in model.parameters():
        p.requires_grad = False

    if scope == "all":
        for p in model.parameters():
            p.requires_grad = True
        return

    if scope == "aux_only":
        for module_name in ("hfold_embedding_model", "hfold_relevancy_model", "hfold_adapters"):
            module = getattr(model, module_name, None)
            if module is not None:
                for p in module.parameters():
                    p.requires_grad = True
        return

    def _enable_backbone_all() -> None:
        if hasattr(model, "gpt_neox"):
            for p in model.gpt_neox.parameters():
                p.requires_grad = True
            if hasattr(model, "embed_out"):
                for p in model.embed_out.parameters():
                    p.requires_grad = True
        elif hasattr(model, "transformer"):
            for p in model.transformer.parameters():
                p.requires_grad = True
            if hasattr(model, "lm_head"):
                for p in model.lm_head.parameters():
                    p.requires_grad = True

    if scope == "backbone_only":
        _enable_backbone_all()
        return

    if scope == "last_k_backbone_plus_aux":
        if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
            layers = model.gpt_neox.layers
            for layer in layers[-int(last_k_layers) :]:
                for p in layer.parameters():
                    p.requires_grad = True
            if hasattr(model.gpt_neox, "final_layer_norm"):
                for p in model.gpt_neox.final_layer_norm.parameters():
                    p.requires_grad = True
            if hasattr(model, "embed_out"):
                for p in model.embed_out.parameters():
                    p.requires_grad = True
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            layers = model.transformer.h
            for layer in layers[-int(last_k_layers) :]:
                for p in layer.parameters():
                    p.requires_grad = True
            if hasattr(model.transformer, "ln_f"):
                for p in model.transformer.ln_f.parameters():
                    p.requires_grad = True
            if hasattr(model, "lm_head"):
                for p in model.lm_head.parameters():
                    p.requires_grad = True
        for module_name in ("hfold_embedding_model", "hfold_relevancy_model", "hfold_adapters"):
            module = getattr(model, module_name, None)
            if module is not None:
                for p in module.parameters():
                    p.requires_grad = True
        return

    raise ValueError(f"unsupported train scope: {scope}")


def _tokenwise_loss(
    *,
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    sliding_window_size: int,
    max_unroll_steps: int,
) -> tuple[torch.Tensor, int]:
    runtime = getattr(model, "hfold_runtime", None)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    total_loss: torch.Tensor | None = None
    total_tokens = 0
    batch_size = int(input_ids.size(0))
    for row in range(batch_size):
        if runtime is not None:
            runtime.reset()
        row_ids = input_ids[row : row + 1]
        row_mask = None if attention_mask is None else attention_mask[row : row + 1]
        seq_len = int(row_ids.size(1))
        max_pred = min(max(1, int(max_unroll_steps)), max(1, seq_len - 1))
        end_pos = min(seq_len, 1 + max_pred)
        for next_pos in range(1, end_pos):
            start_idx = 0
            if sliding_window_size > 0:
                start_idx = max(0, next_pos - int(sliding_window_size))
            prefix_ids = row_ids[:, start_idx:next_pos]
            prefix_mask = None if row_mask is None else row_mask[:, start_idx:next_pos]
            out = model(input_ids=prefix_ids, attention_mask=prefix_mask, use_cache=False)
            logits = out.logits[:, -1, :]
            target = row_ids[:, next_pos]
            nll = F.cross_entropy(logits, target, reduction="mean")
            total_loss = nll if total_loss is None else (total_loss + nll)
            total_tokens += 1
    if total_loss is None:
        raise RuntimeError("no tokens were produced for HFold training loss")
    return total_loss / max(total_tokens, 1), total_tokens


def train_one_epoch(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    sliding_window_size: int,
    max_unroll_steps: int,
    grad_clip: float,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)
        loss, pred_tokens = _tokenwise_loss(
            model=model,
            batch=batch,
            device=device,
            sliding_window_size=sliding_window_size,
            max_unroll_steps=max_unroll_steps,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(grad_clip))
        optimizer.step()
        total_loss += float(loss.item())
        total_tokens += int(pred_tokens)
    avg_loss = total_loss / max(len(dataloader), 1)
    return avg_loss, float(math.exp(avg_loss)) if avg_loss < 50.0 else float("inf")


@torch.no_grad()
def evaluate(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    sliding_window_size: int,
    max_unroll_steps: int,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    for batch in dataloader:
        loss, _ = _tokenwise_loss(
            model=model,
            batch=batch,
            device=device,
            sliding_window_size=sliding_window_size,
            max_unroll_steps=max_unroll_steps,
        )
        total_loss += float(loss.item())
    avg_loss = total_loss / max(len(dataloader), 1)
    return avg_loss, float(math.exp(avg_loss)) if avg_loss < 50.0 else float("inf")


def main() -> None:
    args = parse_args()
    _set_seed(int(args.seed))
    device = _resolve_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = HFoldConfig(
        model=HFoldModelConfig(
            hidden_size=1,  # auto-detected in integration runner
            num_heads=1,
            max_heap_size=int(args.max_heap_size),
            top_w=int(args.max_heap_size),
            pop_k=int(args.max_heap_size),
            aux_fold_interval=max(1, int(args.aux_fold_interval)),
            differentiable_heap=bool(args.differentiable_heap),
            adapter_dim=int(args.adapter_dim),
            embedding_model_type=args.embedding_model_type,
        ),
        training=HFoldTrainingConfig(
            learning_rate=float(args.lr),
            weight_decay=float(args.weight_decay),
            num_epochs=int(args.epochs),
            batch_size=int(args.train_batch_size),
            device=str(device),
            seed=int(args.seed),
        ),
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_loader, eval_loader = _build_lm_dataloaders(args, tokenizer)

    bundle = _build_hfold_bundle(args, cfg)
    model = bundle.model.to(device)
    _set_trainable_scope(model, args.train_scope, int(args.last_k_layers))
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable parameters selected")
    optimizer = torch.optim.AdamW(trainable, lr=float(args.lr), weight_decay=float(args.weight_decay))

    print(
        {
            "device": str(device),
            "train_batches": len(train_loader),
            "eval_batches": len(eval_loader),
            "trainable_params": sum(p.numel() for p in trainable),
            "differentiable_heap": bool(args.differentiable_heap),
            "train_scope": args.train_scope,
        }
    )

    history: list[dict] = []
    for epoch in range(1, int(args.epochs) + 1):
        train_loss, train_ppl = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            sliding_window_size=int(args.sliding_window_size),
            max_unroll_steps=int(args.max_unroll_steps),
            grad_clip=float(args.grad_clip),
        )
        eval_loss, eval_ppl = evaluate(
            model=model,
            dataloader=eval_loader,
            device=device,
            sliding_window_size=int(args.sliding_window_size),
            max_unroll_steps=int(args.max_unroll_steps),
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ppl": train_ppl,
            "eval_loss": eval_loss,
            "eval_ppl": eval_ppl,
        }
        history.append(row)
        print(row)

    # Save backbone checkpoint + tokenizer.
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save HFold aux weights explicitly for reproducible reload.
    if hasattr(model, "hfold_embedding_model"):
        torch.save(model.hfold_embedding_model.state_dict(), os.path.join(args.output_dir, "embedding_autoencoder.pt"))
    if hasattr(model, "hfold_relevancy_model"):
        torch.save(model.hfold_relevancy_model.state_dict(), os.path.join(args.output_dir, "relevancy_transformer.pt"))
    if hasattr(model, "hfold_adapters"):
        torch.save(model.hfold_adapters.state_dict(), os.path.join(args.output_dir, "adapters.pt"))

    with open(os.path.join(args.output_dir, "hfold_finetune_config.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "hfold_config": asdict(cfg), "history": history}, f, indent=2, sort_keys=True)
    print({"saved_to": args.output_dir, "epochs": len(history), "last": history[-1] if history else {}})


if __name__ == "__main__":
    main()
