import math
import time
from pathlib import Path

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from hf_bench.metrics import RuntimeAverages, estimate_decoder_flops, peak_memory_mb
from hf_bench.utils import dump_json
from hf_bench.visualization import save_attention_maps, save_training_curve


class CLMCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
        input_ids = torch.stack(input_ids, dim=0)
        attn_mask = (input_ids != self.tokenizer.pad_token_id).long()
        labels = input_ids.clone()
        return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}


def evaluate(model, loader, device):
    model.eval()
    losses = []
    forward_times = []
    total_tokens = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            t0 = time.perf_counter()
            out = model(**batch)
            forward_times.append(time.perf_counter() - t0)
            losses.append(out.loss.item())
            total_tokens += batch["attention_mask"].sum().item()

    mean_loss = float(sum(losses) / max(1, len(losses)))
    ppl = float(math.exp(min(mean_loss, 20)))
    forward_time = float(sum(forward_times) / max(1, len(forward_times)))
    return {"val_loss": mean_loss, "perplexity": ppl, "forward_time_s": forward_time, "eval_tokens": int(total_tokens)}


def train_one_run(model, tokenizer, tokenized_ds, cfg, run_dir: Path, device: str):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    collator = CLMCollator(tokenizer)
    train_loader = DataLoader(
        tokenized_ds["train"],
        batch_size=cfg.train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        tokenized_ds["validation"],
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=cfg.num_workers,
    )

    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = min(cfg.max_train_steps, len(train_loader) * cfg.epochs)
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    step = 0
    step_times = []
    train_losses = []
    val_losses = []
    global_steps = []
    total_tokens = 0
    step_rows = []

    pbar = tqdm(total=total_steps, desc="train", leave=False)
    optimizer.zero_grad(set_to_none=True)

    for _ in range(cfg.epochs):
        for batch in train_loader:
            if step >= total_steps:
                break

            t0 = time.perf_counter()
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda" and cfg.use_bf16)):
                out = model(**batch)
                loss = out.loss / cfg.grad_accum_steps

            loss.backward()

            if (step + 1) % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            dt = time.perf_counter() - t0
            step_times.append(dt)
            raw_loss = float(loss.item() * cfg.grad_accum_steps)
            train_losses.append(raw_loss)
            step += 1
            global_steps.append(step)

            tokens = int(batch["attention_mask"].sum().item())
            total_tokens += tokens
            step_rows.append({"step": step, "train_loss": raw_loss, "step_time_s": dt, "tokens": tokens})

            pbar.update(1)
            pbar.set_postfix(loss=f"{raw_loss:.4f}")

            if step % cfg.eval_every_steps == 0 or step == total_steps:
                metrics = evaluate(model, val_loader, device)
                val_losses.append(metrics["val_loss"])

        if step >= total_steps:
            break

    pbar.close()

    final_eval = evaluate(model, val_loader, device)

    train_step = float(sum(step_times) / max(1, len(step_times)))
    tokens_per_sec = float(total_tokens / max(1e-9, sum(step_times)))

    sample_batch = next(iter(train_loader))
    seq_len = sample_batch["input_ids"].shape[-1]
    flops = float(estimate_decoder_flops(model, cfg.train_batch_size, seq_len))
    runtime = RuntimeAverages(
        train_step_time_s=train_step,
        forward_time_s=final_eval["forward_time_s"],
        tokens_per_sec=tokens_per_sec,
        peak_memory_mb=peak_memory_mb(),
        flops_per_step=flops,
    )

    attn_batch = next(iter(val_loader))
    attn_batch = {k: v.to(device) for k, v in attn_batch.items()}
    model.eval()
    with torch.no_grad():
        attn_out = model(**attn_batch, output_attentions=True)

    save_attention_maps(attn_out.attentions, run_dir / "attention_maps")
    save_training_curve(global_steps, train_losses, val_losses, run_dir / "training_curve.png")

    model.save_pretrained(run_dir / "checkpoint")
    tokenizer.save_pretrained(run_dir / "checkpoint")

    pd.DataFrame(step_rows).to_csv(run_dir / "step_metrics.csv", index=False)

    payload = {
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_val_loss": final_eval["val_loss"],
        "perplexity": final_eval["perplexity"],
        "runtime": {
            "train_step_time_s": runtime.train_step_time_s,
            "forward_time_s": runtime.forward_time_s,
            "tokens_per_sec": runtime.tokens_per_sec,
            "peak_memory_mb": runtime.peak_memory_mb,
            "flops_per_step": runtime.flops_per_step,
        },
        "history": {
            "step": global_steps,
            "train_loss": train_losses,
            "val_loss": val_losses,
        },
    }
    dump_json(run_dir / "metrics.json", payload)
    return payload
