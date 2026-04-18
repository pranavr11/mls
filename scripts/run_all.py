#!/usr/bin/env python3
import argparse

from hf_bench.config import default_config
from hf_bench.runner import run_experiments


def parse_args():
    p = argparse.ArgumentParser(description="Run HF baseline benchmark suite")
    p.add_argument("--results-root", default="results")
    p.add_argument("--cache-dir", default=".cache")
    p.add_argument("--models", nargs="+", default=["EleutherAI/pythia-160m", "gpt2"])
    p.add_argument("--datasets", nargs="+", default=["pg19", "scrolls"])
    p.add_argument("--scrolls-task", default="gov_report", choices=["gov_report", "qasper"])
    p.add_argument("--seeds", nargs="+", type=int, default=[13, 37, 73, 101])
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-train-steps", type=int, default=300)
    p.add_argument("--eval-every-steps", type=int, default=50)
    p.add_argument("--train-batch-size", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--eval-stride", type=int, default=512)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-bf16", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = default_config()
    cfg.results_root = args.results_root
    cfg.cache_dir = args.cache_dir
    cfg.models = args.models
    cfg.datasets = args.datasets
    cfg.scrolls_task = args.scrolls_task
    cfg.seeds = args.seeds
    cfg.epochs = args.epochs
    cfg.max_train_steps = args.max_train_steps
    cfg.eval_every_steps = args.eval_every_steps
    cfg.train_batch_size = args.train_batch_size
    cfg.eval_batch_size = args.eval_batch_size
    cfg.learning_rate = args.learning_rate
    cfg.weight_decay = args.weight_decay
    cfg.block_size = args.block_size
    cfg.eval_stride = args.eval_stride
    cfg.warmup_ratio = args.warmup_ratio
    cfg.grad_accum_steps = args.grad_accum_steps
    cfg.max_grad_norm = args.max_grad_norm
    cfg.num_workers = args.num_workers
    cfg.use_bf16 = not args.no_bf16

    all_runs, summary = run_experiments(cfg)
    print("Finished experiments")
    print(all_runs)
    print(summary)


if __name__ == "__main__":
    main()
