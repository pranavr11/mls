# HFold Pipeline Human Review Playbook

## Scope
This document summarizes the important HFold pipeline changes since commit `067247a5da62566c4159b07465835136ef86a65f` and proposes the best human review workflow to improve both performance and accuracy.

Included scope:
- HFold runtime/inference modules
- HFold benchmarking + orchestration scripts
- HFold config surface
- HFold model assembly/training paths
- `baselines.ipynb` pipeline-relevant cells

---

## Pipeline Changes That Matter Most

### 1) Tensor top-k heap refactor (core runtime path)
Files:
- `hfold/inference/tensor_heap.py`
- `hfold/inference/hfold_runtime.py`
- `hfold/inference/model_hook.py`

What changed:
- Replaced Python object heap hot path with tensor-native heap bundles and `torch.topk`/gather operations.
- Added tensor pop/reinsert/fold flow and compatibility bridges to legacy entry-style APIs.

Why it matters:
- Directly changes runtime behavior, performance, and tie-order semantics.
- This is the most critical area for output-level regressions and speed differences.

---

### 2) HFold update cadence + scoring behavior controls
Files:
- `hfold/config/schema.py`
- `hfold/inference/model_hook.py`
- `hfold/inference/hfold_runtime.py`

What changed:
- New config knobs: `aux_fold_interval`, `hfold_step_interval`, `candidate_score_mode`, `differentiable_heap`.
- Hook can skip heap updates for non-interval steps.
- Candidate selection can use `attention` or `hidden_dot`.
- Aux fold runs every N timesteps.

Why it matters:
- These knobs directly alter model behavior, quality/speed tradeoffs, and reproducibility.
- Defaults now encode assumptions that must be reviewed deliberately.

---

### 3) HFold benchmark/eval flow changed
Files:
- `hfold/integration/benchmark_runner.py`
- `hfold/scripts/benchmark_all_modes.py`

What changed:
- Added HFold eval controls: `hfold_eval_use_kv_cache` and `hfold_eval_chunk_size`.
- Added broader CLI/config surface for HFold ablations.

Why it matters:
- Benchmark methodology changed; numbers may not be comparable to earlier runs unless config is fixed.

---

### 4) HFold model assembly + aux model variants
Files:
- `hfold/integration/pythia_runner.py`
- `hfold/integration/gpt2_runner.py`
- `hfold/models/embedding_factory.py`
- `hfold/models/lightweight_embedding.py`
- `hfold/models/__init__.py`

What changed:
- Embedding model is now selected via factory (`autoencoder`, `mean_identity`, `mean_bottleneck`).
- Embedding checkpoint loading made permissive (`strict=False`).

Why it matters:
- Enables speed experiments but raises checkpoint compatibility and silent-mismatch risk.

---

### 5) New HFold training/orchestration scripts
Files:
- `hfold/scripts/fine_tune_hfold.py`
- `hfold/scripts/local_wikitext_experiments.py`

What changed:
- Added HFold-aware autoregressive unroll fine-tuning pipeline.
- Added local end-to-end experiment driver for aux extraction/training and benchmark sweeps.

Why it matters:
- New canonical experiment paths with their own assumptions and defaults.

---

### 6) Notebook pipeline updates
File:
- `baselines.ipynb`

What changed:
- Added/expanded aux training/extraction orchestration.
- Added robust hidden-size resolution for aux extraction model config.
- Added explicit benchmark knobs (`HFOLD_EVAL_USE_KV_CACHE`) and passed `aux_fold_interval`.

Why it matters:
- Notebook behavior is now a major source of pipeline outcomes; defaults must align with script defaults.

---

## Best Human Effort: Step-by-Step Algorithm Review

If a human can do one high-value task, do this:

1. Trace one sequence through HFold end-to-end:
   - candidate score
   - top-k pop
   - prepend heap vectors
   - model forward
   - reinsert
   - evict
   - aux summarize/fold
   - next-step state reuse

2. Do this trace in code order:
   - `hfold/inference/model_hook.py`
   - `hfold/inference/hfold_runtime.py`
   - `hfold/inference/tensor_heap.py`
   - `hfold/integration/benchmark_runner.py`

3. For each timestep, log:
   - selected indices/scores
   - heap size + entry ids
   - fold run/skipped state (`aux_fold_interval`)
   - cache mode/chunk setting in effect

Why this is best:
- It simultaneously validates correctness and reveals where runtime cost is concentrated.
- It identifies whether quality regressions come from selection, fold cadence, or eval protocol changes.

---

## High / Medium / Low Impact Buckets

### High-impact
- Tensor heap runtime refactor
- Hook scoring/update cadence changes
- Benchmark semantic changes (`cache`, `chunk`)
- Config/default changes influencing HFold behavior
- Notebook orchestration changes affecting measured outputs

### Medium-impact
- Embedding model factory + lightweight variants
- New fine-tune and local experiment scripts
- Eager attention pinning and checkpoint load behavior

### Low-impact
- Module export surface changes
- Notebook formatting/churn not altering code paths

---

## Human Review Flags (Risk + Verification)

### A) Heap semantics equivalence
Risk:
- `torch.topk` tie behavior/order can differ from Python heap behavior.

Verify:
- Same-seed regression on fixed examples: selected entries, final loss/PPL drift, heap invariants.

### B) Candidate scoring mode
Risk:
- `hidden_dot` and `attention` can produce different candidate sets and quality profiles.

Verify:
- Side-by-side accuracy/speed sweep and pick explicit default for all official runs.

### C) Update cadence knobs
Risk:
- `hfold_step_interval`, `aux_fold_interval`, `hfold_eval_chunk_size` alter algorithm semantics and benchmark comparability.

Verify:
- One-factor sweeps with locked seed/data/checkpoint; declare accepted protocol.

### D) Differentiable heap training path
Risk:
- Graph-connected heap storage changes memory/training dynamics.

Verify:
- Gradient sanity, memory profile, convergence, and inference compatibility of produced checkpoints.

### E) Permissive checkpoint loading (`strict=False`)
Risk:
- Silent missing/unexpected keys.

Verify:
- Log and review load key mismatches; fail when mismatch is unintended.

### F) Notebook/script divergence
Risk:
- Different defaults between notebook and CLI create non-reproducible claims.

Verify:
- Align defaults and record full config in artifacts.

---

## Recommended Human Review Priority Order

1. `model_hook.py` + `hfold_runtime.py` semantic trace (correctness first)
2. `benchmark_runner.py` protocol lock (comparability)
3. `schema.py` defaults and knob policy (reproducibility)
4. `pythia_runner.py` / `gpt2_runner.py` checkpoint compatibility checks
5. `fine_tune_hfold.py` and notebook parity checks

---

## Practical Review Checklist (fast execution)

1. Choose one fixed checkpoint and one fixed eval split slice.
2. Run one-sequence debug trace and inspect heap transitions.
3. Sweep one knob at a time:
   - `candidate_score_mode`
   - `hfold_step_interval`
   - `aux_fold_interval`
   - `hfold_eval_chunk_size`
4. Capture PPL + tok/s + exact config for each run.
5. Lock one Pareto config (speed vs quality) and make it the documented default.
