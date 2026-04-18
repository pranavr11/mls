from pathlib import Path

import pandas as pd
import torch

from hf_bench.config import ExperimentConfig
from hf_bench.data import load_raw_dataset, tokenize_and_chunk
from hf_bench.modeling import load_model_and_tokenizer
from hf_bench.trainer import train_one_run
from hf_bench.utils import device_name, dump_json, ensure_dir, model_tag, set_seed
from hf_bench.visualization import save_summary_plots


def _dataset_tag(dataset_name: str, scrolls_task: str) -> str:
    if dataset_name == "scrolls":
        return f"scrolls__{scrolls_task}"
    return dataset_name


def run_experiments(cfg: ExperimentConfig):
    results_root = ensure_dir(cfg.results_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_rows = []

    for model_name in cfg.models:
        for dataset_name in cfg.datasets:
            ds_tag = _dataset_tag(dataset_name, cfg.scrolls_task)
            raw_ds = load_raw_dataset(dataset_name, cfg.scrolls_task, cfg.cache_dir)

            for seed in cfg.seeds:
                set_seed(seed)
                model, tokenizer = load_model_and_tokenizer(model_name, cfg.cache_dir)
                tokenized = tokenize_and_chunk(dataset_name, raw_ds, tokenizer, cfg.block_size)

                run_dir = results_root / model_tag(model_name) / ds_tag / str(seed)
                ensure_dir(run_dir)

                cfg_payload = cfg.to_dict() | {
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "dataset_variant": ds_tag,
                    "seed": seed,
                    "device": device_name(),
                }
                dump_json(run_dir / "config.json", cfg_payload)

                out = train_one_run(model, tokenizer, tokenized, cfg, run_dir, device)

                row = {
                    "model": model_name,
                    "dataset": ds_tag,
                    "seed": seed,
                    "train_loss": out["final_train_loss"],
                    "val_loss": out["final_val_loss"],
                    "perplexity": out["perplexity"],
                    "train_step_time_s": out["runtime"]["train_step_time_s"],
                    "forward_time_s": out["runtime"]["forward_time_s"],
                    "tokens_per_sec": out["runtime"]["tokens_per_sec"],
                    "flops_per_step": out["runtime"]["flops_per_step"],
                    "peak_memory_mb": out["runtime"]["peak_memory_mb"],
                    "run_dir": str(run_dir),
                }
                all_rows.append(row)

    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(results_root / "all_runs.csv", index=False)

    grouped = (
        all_df.groupby(["model", "dataset"])
        .agg(
            perplexity_mean=("perplexity", "mean"),
            perplexity_std=("perplexity", "std"),
            val_loss_mean=("val_loss", "mean"),
            val_loss_std=("val_loss", "std"),
            train_step_time_mean=("train_step_time_s", "mean"),
            train_step_time_std=("train_step_time_s", "std"),
            forward_time_mean=("forward_time_s", "mean"),
            forward_time_std=("forward_time_s", "std"),
            flops_mean=("flops_per_step", "mean"),
            flops_std=("flops_per_step", "std"),
            memory_mean=("peak_memory_mb", "mean"),
            memory_std=("peak_memory_mb", "std"),
            throughput_mean=("tokens_per_sec", "mean"),
            throughput_std=("tokens_per_sec", "std"),
        )
        .reset_index()
    )

    grouped.to_csv(results_root / "summary_mean_std.csv", index=False)
    dump_json(results_root / "summary_mean_std.json", grouped.to_dict(orient="records"))
    save_summary_plots(grouped, Path(results_root) / "summary_plots")

    return all_df, grouped
