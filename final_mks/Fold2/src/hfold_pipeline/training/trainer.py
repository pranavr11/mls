from __future__ import annotations

import contextlib
import logging
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..config import ExperimentConfig
from ..data.pg19 import load_or_prepare_pg19
from ..modeling.pythia import load_pythia_model_and_tokenizer
from ..utils.io import append_jsonl, ensure_dir, save_json
from .checkpoint import cleanup_old_checkpoints, find_latest_checkpoint, load_checkpoint, save_checkpoint
from .metrics import compute_perplexity, count_tokens_in_batch, get_peak_memory_mb, reset_peak_memory

logger = logging.getLogger(__name__)


class FixedLengthCausalLMCollator:
    def __call__(self, features):
        batch = {}
        for key in features[0].keys():
            batch[key] = torch.tensor([feature[key] for feature in features], dtype=torch.long)
        if "labels" not in batch:
            batch["labels"] = batch["input_ids"].clone()
        return batch


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def autocast_context(training_config, device: torch.device):
    if device.type != "cuda":
        return contextlib.nullcontext()
    if training_config.bf16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if training_config.fp16:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def create_dataloaders(tokenizer, experiment_config: ExperimentConfig):
    datasets = load_or_prepare_pg19(tokenizer, experiment_config.data)
    collator = FixedLengthCausalLMCollator()
    train_dataset = datasets[experiment_config.data.train_split]
    eval_dataset = datasets[experiment_config.data.validation_split]

    train_loader = DataLoader(
        train_dataset,
        batch_size=experiment_config.training.per_device_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=experiment_config.training.dataloader_num_workers,
        pin_memory=experiment_config.training.pin_memory,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=experiment_config.training.per_device_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=experiment_config.training.dataloader_num_workers,
        pin_memory=experiment_config.training.pin_memory,
    )
    return train_loader, eval_loader


def evaluate(
    *,
    model,
    dataloader,
    device: torch.device,
    training_config,
    max_batches: int | None = None,
    profile_throughput: bool = True,
    profile_memory: bool = True,
) -> dict[str, float | int | None]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    total_tokens = 0
    start_time = time.perf_counter()
    if profile_memory:
        reset_peak_memory(device)

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = move_batch_to_device(batch, device)
            with autocast_context(training_config, device):
                outputs = model(**batch)
            total_loss += float(outputs.loss.detach().float().item())
            total_batches += 1
            total_tokens += count_tokens_in_batch(batch)

    elapsed = max(time.perf_counter() - start_time, 1e-6) if profile_throughput else None
    avg_loss = total_loss / max(total_batches, 1)
    metrics = {
        "loss": avg_loss,
        "perplexity": compute_perplexity(avg_loss),
        "tokens": total_tokens,
        "tokens_per_second": (total_tokens / elapsed) if profile_throughput else None,
        "peak_memory_mb": get_peak_memory_mb(device) if profile_memory else None,
        "num_batches": total_batches,
    }
    model.train()
    return metrics


def train(experiment_config: ExperimentConfig) -> dict[str, float | int | None]:
    from transformers import get_scheduler, set_seed

    ensure_dir(experiment_config.training.output_dir)
    metrics_path = Path(experiment_config.training.output_dir) / "metrics.jsonl"
    summary_path = Path(experiment_config.training.output_dir) / "summary.json"

    set_seed(experiment_config.runtime.seed)
    device = resolve_device(experiment_config.runtime.device)
    model, tokenizer = load_pythia_model_and_tokenizer(experiment_config)
    train_loader, eval_loader = create_dataloaders(tokenizer, experiment_config)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=experiment_config.training.learning_rate,
        betas=(experiment_config.training.adam_beta1, experiment_config.training.adam_beta2),
        eps=experiment_config.training.adam_epsilon,
        weight_decay=experiment_config.training.weight_decay,
    )

    updates_per_epoch = math.ceil(
        len(train_loader) / experiment_config.training.gradient_accumulation_steps
    )
    total_steps = (
        experiment_config.training.max_steps
        if experiment_config.training.max_steps > 0
        else math.ceil(experiment_config.training.num_train_epochs * updates_per_epoch)
    )
    warmup_steps = int(total_steps * experiment_config.training.warmup_ratio)

    scheduler = get_scheduler(
        experiment_config.training.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = None
    if device.type == "cuda" and experiment_config.training.fp16:
        scaler = torch.cuda.amp.GradScaler()

    start_epoch = 0
    global_step = 0
    resume_target = experiment_config.training.resume_from_checkpoint
    if resume_target is None:
        latest = find_latest_checkpoint(experiment_config.training.output_dir)
        if latest is not None:
            resume_target = str(latest)

    if resume_target:
        logger.info("Resuming training from %s", resume_target)
        trainer_state = load_checkpoint(
            checkpoint_dir=resume_target,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        global_step = int(trainer_state.get("global_step", 0))
        start_epoch = int(trainer_state.get("epoch", 0))

    if global_step >= total_steps:
        logger.info("Checkpoint already reached requested total_steps=%d; running final eval only.", total_steps)
        final_metrics = evaluate(
            model=model,
            dataloader=eval_loader,
            device=device,
            training_config=experiment_config.training,
        )
        save_json(final_metrics, summary_path)
        return final_metrics

    logger.info(
        "Starting training: attention=%s, total_steps=%d, block_size=%d",
        experiment_config.attention.attention_type,
        total_steps,
        experiment_config.data.block_size,
    )

    running_loss = 0.0
    running_updates = 0
    running_tokens = 0
    interval_start = time.perf_counter()
    progress = tqdm(total=total_steps, initial=global_step, desc="Training", dynamic_ncols=True)
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, math.ceil(experiment_config.training.num_train_epochs)):
        if global_step >= total_steps:
            break

        for micro_step, batch in enumerate(train_loader, start=1):
            if global_step >= total_steps:
                break

            batch = move_batch_to_device(batch, device)
            batch_tokens = count_tokens_in_batch(batch)

            with autocast_context(experiment_config.training, device):
                outputs = model(**batch)
                loss = outputs.loss
                scaled_loss = loss / experiment_config.training.gradient_accumulation_steps

            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            running_loss += float(loss.detach().float().item())
            running_tokens += batch_tokens

            should_update = (
                micro_step % experiment_config.training.gradient_accumulation_steps == 0
                or micro_step == len(train_loader)
            )
            if not should_update:
                continue

            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                experiment_config.training.max_grad_norm,
            )
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            running_updates += 1
            progress.update(1)

            if global_step % experiment_config.training.log_interval == 0:
                elapsed = max(time.perf_counter() - interval_start, 1e-6)
                train_metrics = {
                    "event": "train",
                    "epoch": epoch,
                    "global_step": global_step,
                    "loss": running_loss / max(running_updates, 1),
                    "perplexity": compute_perplexity(running_loss / max(running_updates, 1)),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "tokens_per_second": running_tokens / elapsed,
                    "peak_memory_mb": get_peak_memory_mb(device),
                }
                append_jsonl(train_metrics, metrics_path)
                logger.info(
                    "step=%d loss=%.4f ppl=%.2f tok/s=%.1f mem=%sMB",
                    global_step,
                    train_metrics["loss"],
                    train_metrics["perplexity"],
                    train_metrics["tokens_per_second"],
                    f"{train_metrics['peak_memory_mb']:.1f}" if train_metrics["peak_memory_mb"] else "n/a",
                )
                running_loss = 0.0
                running_updates = 0
                running_tokens = 0
                interval_start = time.perf_counter()
                reset_peak_memory(device)

            if global_step % experiment_config.training.eval_interval == 0:
                eval_metrics = evaluate(
                    model=model,
                    dataloader=eval_loader,
                    device=device,
                    training_config=experiment_config.training,
                )
                eval_record = {"event": "eval", "global_step": global_step, **eval_metrics}
                append_jsonl(eval_record, metrics_path)
                logger.info(
                    "eval step=%d loss=%.4f ppl=%.2f tok/s=%.1f",
                    global_step,
                    eval_metrics["loss"],
                    eval_metrics["perplexity"],
                    eval_metrics["tokens_per_second"],
                )

            if global_step % experiment_config.training.save_interval == 0:
                checkpoint_dir = Path(experiment_config.training.output_dir) / f"checkpoint-{global_step:07d}"
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    experiment_config=experiment_config,
                    trainer_state={"global_step": global_step, "epoch": epoch},
                )
                cleanup_old_checkpoints(
                    experiment_config.training.output_dir,
                    experiment_config.training.max_checkpoints,
                )

    progress.close()

    final_metrics = evaluate(
        model=model,
        dataloader=eval_loader,
        device=device,
        training_config=experiment_config.training,
    )
    final_record = {"event": "final_eval", "global_step": global_step, **final_metrics}
    append_jsonl(final_record, metrics_path)
    save_json(final_record, summary_path)

    save_checkpoint(
        checkpoint_dir=Path(experiment_config.training.output_dir) / "checkpoint-final",
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        experiment_config=experiment_config,
        trainer_state={"global_step": global_step, "epoch": epoch},
    )

    return final_record


def evaluate_checkpoint(
    experiment_config: ExperimentConfig,
    checkpoint_path: str | None = None,
    *,
    split: str | None = None,
    max_batches: int | None = None,
    profile_throughput: bool = True,
    profile_memory: bool = True,
) -> dict[str, float | int | None]:
    device = resolve_device(experiment_config.runtime.device)
    model, tokenizer = load_pythia_model_and_tokenizer(experiment_config)
    model.to(device)

    if checkpoint_path:
        load_checkpoint(checkpoint_dir=checkpoint_path, model=model, device=device)

    datasets = load_or_prepare_pg19(tokenizer, experiment_config.data)
    split_name = split or experiment_config.data.validation_split
    collator = FixedLengthCausalLMCollator()
    dataloader = DataLoader(
        datasets[split_name],
        batch_size=experiment_config.training.per_device_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=experiment_config.training.dataloader_num_workers,
        pin_memory=experiment_config.training.pin_memory,
    )

    return evaluate(
        model=model,
        dataloader=dataloader,
        device=device,
        training_config=experiment_config.training,
        max_batches=max_batches,
        profile_throughput=profile_throughput,
        profile_memory=profile_memory,
    )
