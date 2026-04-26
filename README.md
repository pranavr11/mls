# MLSFINAL — HFold

15-442 (Machine Learning Systems) final project. Implements the **HFold**
(heap-fold) inference-time algorithm on top of fine-tuned sliding-window
backbones (Pythia, GPT-2). See `HFOLD_PIPELINE_PLAN.md` for the canonical
algorithm spec and design contract.

## Project layout

```
hfold/
  config/                     # typed config (HFoldConfig / HFoldModelConfig / HFoldTrainingConfig)
  data/                       # real shard dataset + extractor + collate
  inference/                  # heap, runtime, model_hook, vector_store
  integration/                # benchmark + per-backbone HFold builders
  models/                     # adapters, embedding autoencoder, relevancy transformer
  scripts/                    # CLIs (extract / train_aux / benchmark / generate)
  tests/                      # pytest suite (32 tests at last count)
  training/                   # train_embedding, train_relevancy, train_joint_aux, losses, metrics
fine_tune.py                  # vanilla fine-tune (full attention)
new_fine_tune.py              # sliding-window fine-tune
HFOLD_PIPELINE_PLAN.md        # design doc & spec
```

## End-to-end pipeline

The pipeline is four commands, each idempotent. All scripts have `--help`.

### 1. Sliding-window fine-tune the backbone(s)

Use the existing trainer; output checkpoints land in `./checkpoints/`.

```bash
# Pythia (default in CONFIG)
python new_fine_tune.py
# Edit `CONFIG['model_name'] = 'gpt2'` and re-run for GPT-2.
python new_fine_tune.py
```

Outputs: `./checkpoints/full_finetuning/` (rename per backbone, e.g.
`pythia_full_finetuning/`, `gpt2_full_finetuning/`).

### 2. Extract real hidden-state shards

```bash
python -m hfold.scripts.extract_hidden_states \
  --backbone pythia \
  --model-name EleutherAI/pythia-31m \
  --checkpoint-dir ./checkpoints/pythia_full_finetuning \
  --dataset wikitext \
  --output-dir ./data/extracted/pythia \
  --max-chunks 128

python -m hfold.scripts.extract_hidden_states \
  --backbone gpt2 \
  --model-name gpt2 \
  --checkpoint-dir ./checkpoints/gpt2_full_finetuning \
  --dataset wikitext \
  --output-dir ./data/extracted/gpt2 \
  --max-chunks 128
```

Each shard is a list of dicts:
`{backbone, heap_vectors[S, hidden], evicted_vectors[S, hidden], teacher_scores[S]}`
where `teacher_scores` is the actual softmaxed attention probabilities of an
anchor token over its top-S keys. No synthetic placeholders.

### 3. Train the embedding + relevancy aux models

```bash
python -m hfold.scripts.train_aux_models \
  --extracted-dirs ./data/extracted/pythia ./data/extracted/gpt2 \
  --backbone-dims pythia=256 gpt2=768 \
  --output-dir ./checkpoints/aux \
  --epochs 3
```

Outputs:

- `./checkpoints/aux/embedding_autoencoder.pt`
- `./checkpoints/aux/relevancy_transformer.pt`
- `./checkpoints/aux/adapters.pt`

### 4. Run all three benchmarks for both backbones

```bash
python -m hfold.scripts.benchmark_all_modes \
  --backbone pythia \
  --model-name EleutherAI/pythia-31m \
  --checkpoint-dir ./checkpoints/pythia_full_finetuning \
  --aux-dir ./checkpoints/aux \
  --dataset wikitext

python -m hfold.scripts.benchmark_all_modes \
  --backbone gpt2 \
  --model-name gpt2 \
  --checkpoint-dir ./checkpoints/gpt2_full_finetuning \
  --aux-dir ./checkpoints/aux \
  --dataset wikitext
```

Each prints three rows: `full_attention | sliding_window | hfold` with
`loss`, `perplexity`, and `tokens/sec`. The benchmark refuses to run HFold
with random aux weights unless you explicitly pass `--allow-random-aux`.

### Optional — generate text with HFold

```bash
python -m hfold.scripts.run_hfold_inference \
  --backbone pythia \
  --model-name EleutherAI/pythia-31m \
  --checkpoint ./checkpoints/pythia_full_finetuning \
  --embedding-checkpoint ./checkpoints/aux/embedding_autoencoder.pt \
  --relevancy-checkpoint ./checkpoints/aux/relevancy_transformer.pt \
  --adapters-checkpoint ./checkpoints/aux/adapters.pt \
  --prompt "Once upon a time"
```

## Tests

```bash
PYTHONPATH=. pytest -q hfold/tests
```

Currently 32 tests across heap correctness, runtime semantics, fold equation,
attention-mask expansion, multi-backbone training, real shard extraction,
checkpoint loading, KV-cache compatibility, and benchmark runner reset
behavior.

## Notes

- HFold inference disables KV caching (`use_cache=False`). Heap rows have no
  fixed past-token positions, so caching is incompatible with prepend-based
  injection. This is documented in `hfold/inference/model_hook.py`.
- The aux models share latent space across Pythia and GPT-2 via
  `BackboneAdapterRegistry` so a single trained pair generalizes to both.
- The runtime is reset between independent eval sequences in
  `_run_eval`; heap state never leaks across batches.
