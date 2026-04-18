# HF Baseline Benchmark Pipeline (PG-19 + SCROLLS)

Reproducible benchmarking pipeline for baseline full-attention transformers:
- `EleutherAI/pythia-160m`
- `gpt2`

Datasets:
- `pg19`
- `tau/scrolls` (`gov_report` by default, supports `qasper`)

The pipeline fine-tunes each `(model, dataset, seed)` run, logs quality + efficiency metrics, saves checkpoints, attention maps, training curves, and aggregate summary tables.

## What It Logs
Per run:
- training loss
- validation loss
- perplexity
- forward-pass latency
- training-step latency
- throughput (tokens/sec)
- FLOPs/step (decoder-LM approximation)
- peak GPU memory

Artifacts are stored under:
- `results/{model}/{dataset}/{seed}/`

Global summaries:
- `results/all_runs.csv`
- `results/summary_mean_std.csv`
- `results/summary_mean_std.json`
- `results/summary_plots/*.png`

## Project Layout
- `hf_bench/data.py` dataset loading + tokenization/chunking
- `hf_bench/modeling.py` model/tokenizer loading
- `hf_bench/trainer.py` training + evaluation loops + metric logging
- `hf_bench/metrics.py` runtime/FLOPs/memory metrics
- `hf_bench/visualization.py` attention maps + curves + summary plots
- `hf_bench/runner.py` experiment matrix orchestration and aggregation
- `scripts/run_all.py` single CLI entrypoint
- `scripts/colab_run.sh` Colab-friendly one-command launch

## Local Setup (Conda)
```bash
conda env create -f environment.yml
conda activate hf-bench-15442
python -m pip install -e .
```

Run full benchmark matrix:
```bash
python scripts/run_all.py \
  --models EleutherAI/pythia-160m gpt2 \
  --datasets pg19 scrolls \
  --scrolls-task gov_report \
  --seeds 13 37 73 101 \
  --max-train-steps 300
```

## Colab Setup (GPU)
In a Colab cell:
```bash
!git clone <your-repo-url>
%cd FINALPROJ
!bash scripts/colab_run.sh
```

Or run manually in Colab:
```bash
!pip install -e .
!python scripts/run_all.py --max-train-steps 200 --seeds 13 37 73
```

## Reproducibility Notes
- Seeds are fixed for Python, NumPy, and PyTorch.
- cuDNN deterministic mode is enabled where possible.
- Some GPU kernels may still introduce minor non-determinism depending on CUDA/PyTorch versions.

## Example Outputs per Run
- `config.json`
- `metrics.json`
- `step_metrics.csv`
- `training_curve.png`
- `attention_maps/*.png`
- `checkpoint/*`

## CLI Options
```bash
python scripts/run_all.py --help
```
