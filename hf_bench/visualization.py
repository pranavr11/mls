from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_training_curve(steps: List[int], train_losses: List[float], val_losses: List[float], out_path: Path):
    if not steps or not train_losses:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(steps[: len(train_losses)], train_losses, label="train_loss")
    if val_losses:
        eval_steps = np.linspace(steps[0], steps[-1], len(val_losses), dtype=int)
        plt.plot(eval_steps, val_losses, label="val_loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_attention_maps(attentions, out_dir: Path, layer_ids: List[int] | None = None, head_ids: List[int] | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    n_layers = len(attentions)
    layer_ids = layer_ids if layer_ids is not None else [0, n_layers // 2, n_layers - 1]

    for layer in layer_ids:
        layer_attn = attentions[layer][0].detach().float().cpu().numpy()
        n_heads = layer_attn.shape[0]
        heads = head_ids if head_ids is not None else [0, min(1, n_heads - 1), n_heads - 1]

        for head in heads:
            mat = layer_attn[head]
            plt.figure(figsize=(5, 4))
            plt.imshow(mat, aspect="auto", cmap="viridis")
            plt.colorbar()
            plt.title(f"Layer {layer} Head {head}")
            plt.xlabel("Key Position")
            plt.ylabel("Query Position")
            plt.tight_layout()
            plt.savefig(out_dir / f"attn_layer{layer}_head{head}.png")
            plt.close()


def save_summary_plots(summary_df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("perplexity_mean", "perplexity_std", "Perplexity"),
        ("train_step_time_mean", "train_step_time_std", "Train Step Time (s)"),
        ("throughput_mean", "throughput_std", "Tokens / sec"),
        ("flops_mean", "flops_std", "FLOPs / step"),
        ("memory_mean", "memory_std", "Peak Memory (MB)"),
    ]

    labels = [f"{m}\n{d}" for m, d in zip(summary_df["model"], summary_df["dataset"])]

    for mean_col, std_col, title in metrics:
        plt.figure(figsize=(10, 5))
        x = np.arange(len(labels))
        y = summary_df[mean_col].to_numpy()
        yerr = summary_df[std_col].fillna(0).to_numpy()
        plt.bar(x, y, yerr=yerr, capsize=4)
        plt.xticks(x, labels, rotation=20, ha="right")
        plt.title(title)
        plt.tight_layout()
        fname = title.lower().replace(" ", "_").replace("/", "_") + ".png"
        plt.savefig(out_dir / fname)
        plt.close()
