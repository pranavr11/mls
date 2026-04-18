#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export MPLCONFIGDIR="$(pwd)/.cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python scripts/run_all.py \
  --results-root results \
  --cache-dir .cache \
  --models EleutherAI/pythia-160m gpt2 \
  --datasets pg19 scrolls \
  --scrolls-task gov_report \
  --seeds 13 37 73 \
  --epochs 1 \
  --max-train-steps 200 \
  --eval-every-steps 50 \
  --train-batch-size 1 \
  --eval-batch-size 1 \
  --block-size 1024
