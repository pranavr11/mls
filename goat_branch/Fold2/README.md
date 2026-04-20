# HFold Pipeline

This repository is a controlled ablation framework for comparing attention mechanisms on top of the same pretrained backbone:

- model: `EleutherAI/pythia-160m`
- task: causal language modeling fine-tuning
- dataset: `emozilla/pg19`
- variable under study: attention implementation

The pipeline is built so the only intended architectural change is the attention path. Everything else stays fixed unless the config says otherwise.

## Supported attention modes

- `full`: original pretrained Pythia attention
- `sliding_window`: local causal attention with a fixed backward window
- `hfold`: in-tree HFold integration using the merged HFold implementation
- aliases: `self_attention` maps to `full`, and `sliding_attention` maps to `sliding_window`

HFold now ships in-tree as the default backend for `attention.attention_type: hfold`. The external backend hook is still available if we want to swap in a different implementation later, but the training, evaluation, and benchmarking pipeline can run HFold directly without extra wiring.

## Repo layout

```text
configs/
  full.yaml
  sliding_window.yaml
  hfold.yaml
src/hfold_pipeline/
  attention/
  data/
  modeling/
  training/
  utils/
  scripts/
tests/
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Typical workflow

1. Preprocess PG-19 into fixed-length language-modeling blocks.
2. Train a run with one attention backend.
3. Evaluate and benchmark.

```bash
hfold-prepare-pg19 --config configs/pythia_pg19_sliding.yaml
hfold-train --config configs/pythia_pg19_sliding.yaml
hfold-eval --config configs/pythia_pg19_sliding.yaml --checkpoint artifacts/runs/sliding_window/checkpoint-final
hfold-benchmark --config configs/pythia_pg19_sliding.yaml --checkpoint artifacts/runs/sliding_window/checkpoint-final
```

You can also override the attention mechanism from the CLI without editing the YAML:

```bash
hfold-train --config configs/pythia_pg19_full.yaml --attention-type self_attention
hfold-train --config configs/pythia_pg19_full.yaml --attention-type sliding_attention
hfold-train --config configs/pythia_pg19_full.yaml --attention-type hfold
```

## HFold backend contract

When `attention.attention_type: hfold`, the pipeline will use the native in-tree HFold backend by default. Optionally, `attention.hfold_backend` can still point to an importable callable using the format:

```text
package.module:callable_name
```

The callable receives:

- `base_attention`: the patched GPT-NeoX attention module
- `original_forward`: the original bound forward method
- `bound_arguments`: the current forward arguments as a mutable mapping
- `sliding_window_attention_mask`: the local causal mask already built for the current step
- `layer_index`: transformer layer index
- `hfold_config`: parsed HFold config block

That lets HFold reuse the same pretrained QKV projections and the same training pipeline.

## Design notes

- Pythia-160M is loaded through Hugging Face `transformers`.
- Attention patching is done in place so pretrained weight loading and checkpoint keys remain compatible.
- The config forces `eager` attention by default so custom additive masks are respected during ablations.
- HFold-specific fold/retrieval parameters are registered as submodules on each patched attention layer, so they train and checkpoint with the rest of the model.
- Training logs include loss, perplexity, throughput, and CUDA memory when available.
- Benchmark output now includes eval/inference throughput, optional eval memory profiling, and analytical FLOPs estimates for the active attention backend and the full transformer forward pass.
