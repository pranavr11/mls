# HFold End-to-End Pipeline Plan (v2)

This document is the implementation contract before any code is written. Once approved, the codebase will exactly match this plan — no placeholders, no synthetic data substitutes, and the user-facing UX is "run a few commands and get the three-mode benchmark numbers for both Pythia and GPT-2."

---

## 1. Scope

We commit to four deliverables:

1. **Algorithm**: a faithful implementation of HFold (heap folding) at inference time on top of an already-fine-tuned sliding-window model.
2. **Aux models**: an embedding autoencoder and a relevancy transformer, with real training data, real loss surfaces, and saved checkpoints.
3. **Integration**: HFold wraps a fine-tuned sliding-window model (Pythia and GPT-2). The benchmark contrasts three modes on the same fine-tuned weights:
   - full attention (vanilla pretrained backbone),
   - sliding window (already-fine-tuned),
   - HFold (sliding-window weights + heap fold runtime + trained aux models).
4. **UX**: 4 single-line commands take a developer from raw weights to printed benchmark numbers.

Everything lives under `hfold/` (and a thin top-level `scripts/` wrapper). The directory `final_mks/` is left untouched (it’s an old prototype).

---

## 2. Algorithm Specification (Canonical)

We follow your prompt verbatim. The only design choices the prompt left open are flagged here as **DECISION**.

### 2.1 State

- `W` (sliding window size; e.g. 256) — already used by `new_fine_tune.py`.
- `S` (heap capacity; e.g. 16).
- `K` (pop count per timestep; e.g. 8).
- `w` (top attention candidates per timestep; e.g. 8).
- One **global** max-heap of `(score, vector, metadata)` shared across all timesteps and all decoder layers. This matches "the heap is global across all time steps".

### 2.2 What is a "timestep"?

The prompt says HFold is for autoregressive decoders. A literal "one HFold step per generated token" is correct but extremely slow and breaks batched perplexity evaluation.

**DECISION: chunked autoregressive HFold for benchmark eval, per-token HFold for `generate()`.**

- For **benchmark eval (perplexity / loss)**: split each evaluation example into non-overlapping chunks of length `W`. Each chunk = one timestep. Heap state persists across chunks within the same sequence and is reset between sequences. This keeps the algorithm exactly as written (`timestep 0`, `timestep t`) without per-token cost.
- For **generation (`model.generate`)**: each call to the trunk’s `forward` is one timestep, which is naturally per-token under HF's generate loop with HFold (since we force `use_cache=False`, every step re-encodes; but the heap advances once per token, which is the correct algorithmic semantics).

This decision is documented in code (`hfold/inference/model_hook.py`) and in the README. We will write an explicit test for both modes (`test_chunked_eval_advances_per_chunk`, `test_generate_advances_per_token`).

### 2.3 Timestep 0

1. Forward pass through the entire sliding-window backbone on the input chunk (or first prompt) with `output_attentions=True`, `output_hidden_states=False` (we only need the last layer for top-w selection — see DECISION below).
2. From the **last-layer attention map**, compute `score_per_key = mean over heads, mean over query rows` (mean-of-means over heads/queries, restricted to keys that are real tokens).
3. Take the top-`w` keys. Their last-layer hidden vectors are the candidate vectors.
4. Push `w` entries into the heap. (No retrieval on timestep 0.)

**DECISION: last-layer scope.** The prompt says "after passing through all the layers"; we interpret "the attention scores" as the last layer’s attention map. (We will optionally support a layer-MoE relevancy head later, see §4.3.)

### 2.4 Timestep `t > 0`

1. Pop top-`K` entries from the heap. Their stored `vector` field is the post-attention transformed vector from a previous timestep.
2. **Prepend** these `K` vectors to the current chunk’s embedded inputs (so length becomes `chunk_len + K`).
3. Forward through the entire backbone. The last layer's output gives both transformed token rows and transformed heap rows.
4. Reinsertion candidates:
   - the K transformed-popped vectors (post-attention versions),
   - the top-`w` highest-attention vectors at this timestep, **excluding** any whose token position equals one of the K popped positions (de-duplication).
5. Push reinsertion candidates into the heap → some entries may be evicted because heap is bounded to `S`.
6. The evicted set `E` (size `≤ S`) is sent through the **embedding autoencoder**, producing one summary vector `g`.
7. Retrieve all current heap vectors `h_1..h_S` (no pop). The **relevancy transformer** outputs `r_i = Rel(g, h_i)` for each `i`.
8. **Fold update**: `h_i ← h_i + r_i · g` for every entry in the heap.
9. Continue.

The runtime maintains *one* global heap; the prompt says "the heap is global across all time steps", so we do **not** keep per-layer heaps (the existing `per_layer_heap=True` config flag is removed; we always use a single global heap).

### 2.5 Where heap vectors live in attention

We prepend the K heap vectors at the front of the embed sequence. Combined with the standard causal mask, this means original tokens at position `i` can attend to all heap rows (positions `0..K-1`) plus their own causal context. This matches "append this to the inputs for this timestep". We expand the attention mask correctly for both 2-D and 4-D HF mask shapes (already implemented and tested).

### 2.6 Cache compatibility

KV caching is incompatible with our prepend-based heap injection because heap rows have no fixed past-token positions. **DECISION: we force `use_cache=False` inside the HFold trunk hook**, run a single augmented forward per timestep, and accept the slowdown. Generation works (HF's generate loop falls back to re-encoding). This was already validated with logs in the prior debugging round.

---

## 3. Embedding Model (Autoencoder)

### 3.1 Architecture (`hfold/models/embedding_autoencoder.py`)

We keep the current shape-correct implementation but state it precisely:

- Input: `[batch, S, shared_dim]` (vectors are pre-mapped to shared latent dim through `BackboneAdapterRegistry`; padded with a learned padding vector if fewer than `S` evicted vectors are available).
- Encoder: `(Linear → GELU) × num_layers`, then mean-pool over the S slots, then `Linear(shared_dim → latent_dim)`, then `LayerNorm(latent_dim)`. Output is the bottleneck `g ∈ [batch, latent_dim]`.
- Decoder (slot-aware, already in place): for each slot `i`, concatenate `g` with a learnable per-slot query `q_i`, run a 2-layer MLP, output `shared_dim`. Reconstruction `R ∈ [batch, S, shared_dim]`.
- Loss: cosine reconstruction loss + L2 stabilization. Padded slots are masked out.

### 3.2 Padding strategy

**DECISION**: when `|E| < S`, pad with the learned padding vector (config flag `pad_token_strategy="learned"` becomes the default). This avoids zero-vector pathologies in cosine similarity.

### 3.3 Training data (real)

`hfold/data/extract_hidden_states.py` (new):

- Loads a fine-tuned sliding-window checkpoint (Pythia or GPT-2).
- Streams a long-text corpus (WikiText-103 by default — same as `fine_tune.py`; SCROLLS as alt).
- Tokenizes into chunks of length `chunk_len` (e.g. 512).
- For each chunk, runs `model(... output_hidden_states=True, output_attentions=True, use_cache=False)`.
- For several anchor positions per chunk (`num_anchors`, e.g. 4), takes:
  - the last-layer hidden states at all key positions `< anchor`,
  - the anchor row of the last-layer attention map (mean over heads), softmaxed.
- For each anchor, builds one training tuple:
  - **heap_vectors**: hidden vectors of the top-`S` keys by attention.
  - **evicted_vectors**: hidden vectors of the next `S` keys (positions S..2S-1 by attention rank). These represent "what would get evicted next step".
  - **teacher_scores**: the actual attention probabilities over the top-S heap_vectors (renormalized).
  - **backbone tag**: "pythia" or "gpt2".
- Saves shards as `*.pt` files in `data/extracted/<backbone>/shard_XXXX.pt`.

### 3.4 Embedding training (real)

`hfold/training/train_embedding.py` (rewrite):

- Combined dataset across backbones (mixing Pythia and GPT-2 shards). Each example carries its backbone tag.
- Per-row encode through `BackboneAdapter[backbone].to_shared` into shared latent.
- Forward through the autoencoder. Decoder output is per-row decoded back through `BackboneAdapter[backbone].from_shared`.
- Loss: cosine reconstruction in **original backbone space** (so the adapter pair must invert cleanly) — masked over padded slots.
- Optimizer: AdamW. Cosine LR schedule. Gradient clipping.

### 3.5 What it learns

A single embedding model + adapter registry that compresses up to S real evicted hidden states from either backbone into one summary vector and reconstructs them well. This is exactly what the algorithm asks for.

---

## 4. Relevancy Model

### 4.1 Architecture (`hfold/models/relevancy_transformer.py`)

Already shape-correct. State it precisely:

- Input: `[batch, 1+S, latent_dim]` formed by prepending `g` to the heap in latent space.
- Encoder-only transformer (4 layers, 4 heads, GELU, dropout).
- Score head reads `[encoded_g, encoded_h_i]` and emits one scalar per heap slot.
- Output: `[batch, S]` raw scores. The HFold runtime applies them as `r_i` (no softmax — per the prompt, `h_i + r_i · g` uses the raw relevancy score; we keep raw scores at inference but compare softmax distributions to teacher attention distributions during training).

### 4.2 Training target

The teacher distribution is the softmax attention probabilities from the anchor token over the S heap_vectors (already extracted in §3.3). Loss is a weighted sum of:

- KL(softmax(pred) ∥ teacher),
- MSE on the softmax distributions,
- pairwise ranking loss (preserves top-1 ordering).

### 4.3 Optional layer-MoE (future, marked **OUT OF SCOPE for v1**)

The prompt mentions "maybe we can do MoE across layers as well". For the first cut, we extract from the last layer only. The architecture leaves room for a layer-MoE head later but we will not ship it in v1 to avoid scope creep.

### 4.4 Universality across backbones

The relevancy model is shared across Pythia and GPT-2 by working in the shared latent space (via the same `BackboneAdapterRegistry`). This is what makes "we can use this universally between Pythia and GPT-2" actually true.

---

## 5. Integration with the Sliding-Window Fine-Tune

### 5.1 Reused as-is

- `new_fine_tune.py`: produces sliding-window-fine-tuned checkpoints for Pythia and GPT-2 (already does both). We do **not** rewrite this; we make sure the rest of the system reads its `./checkpoints/full_finetuning/` outputs.

### 5.2 Updated builders

- `hfold/integration/pythia_runner.py` and `hfold/integration/gpt2_runner.py`:
  - Already accept `embedding_checkpoint_path`, `relevancy_checkpoint_path`, `adapters_checkpoint_path` (added in the prior round).
  - We update them so the **default** when those are not provided is to look in `./checkpoints/aux/{embedding,relevancy,adapters}.pt`. Falling back to random weights raises a loud warning instead of silently using random weights — because that was the worst part of the original codebase.

### 5.3 Benchmark runner

`hfold/integration/benchmark_runner.py`:

- `_run_eval` already resets HFold runtime per batch.
- HFold mode runs autoregressively but with a **bounded context window** of size `W` (`hfold_window_size=sliding_window_size`), so each timestep forward is capped to at most `W` tokens of real sequence input while the global heap state persists across timesteps. This keeps end-to-end HFold benchmark cost linear in sequence length for fixed `W` (matching sliding-window spirit) instead of triangular-prefix `O(n^2)` reprocessing.
- The HFold model hook keeps the backbone's standard sliding-window KV cache **on**: it does NOT force `use_cache=False` and does NOT strip caller-supplied `past_key_values`. The K heap rows are prepended only as inputs for THIS timestep, and the hook splices them back out of the returned `past_key_values` so future timesteps see a clean sequence-only cache (no heap pollution). `position_ids` is dropped on the augmented call so the backbone re-derives positions from `past_kv_length + current_input_length` consistent with the prepended heap rows + the real new tokens.
- The full-attention and sliding-window modes continue to do single-pass forward (they don't need chunking).
- All three modes share the same dataloader, tokenizer, and chunk boundary.

### 5.4 Real eval datasets

Use the existing dataloader path from `new_fine_tune.py` (`build_dataloaders`) so the eval dataset is the same one used to fine-tune the sliding-window model: WikiText-103 (default), SCROLLS GovReport (optional).

`hfold/scripts/benchmark_all_modes.py` is rewired so it loads the real eval dataloader instead of the synthetic dummy one. The synthetic dataset and `_dummy_lm_collate` are deleted.

---

## 6. Final File Plan (after this work)

```
hfold/
  __init__.py
  _debug_log.py                      # delete (debug helper, no longer needed)
  config/
    schema.py                        # remove per_layer_heap; add chunk_len, num_anchors, etc.
  inference/
    heap_state.py                    # keep
    priority_heap.py                 # keep
    vector_store.py                  # keep
    hfold_runtime.py                 # keep (with global-only mode)
    model_hook.py                    # keep (single augmented forward, use_cache=False)
  models/
    interfaces.py                    # keep
    adapters.py                      # keep
    embedding_autoencoder.py         # keep (slot-aware decoder, learned pad)
    relevancy_transformer.py         # keep
  data/
    extract_hidden_states.py         # NEW (real extractor)
    hidden_state_dataset.py          # REWRITE: HiddenStateShardDataset reads .pt shards
    collate.py                       # keep (already shape-correct)
  training/
    losses.py                        # keep
    metrics.py                       # keep
    train_embedding.py               # REWRITE: real shards, save state_dict
    train_relevancy.py               # REWRITE: real shards, save state_dict
    train_joint_aux.py               # keep (orchestrates the two trainers)
  integration/
    pythia_runner.py                 # keep + sane defaults for aux paths
    gpt2_runner.py                   # keep + sane defaults for aux paths
    benchmark_runner.py              # ADD chunked HFold eval
  scripts/
    extract_hidden_states.py         # NEW CLI
    train_aux_models.py              # NEW CLI (replaces the two split scripts)
    benchmark_all_modes.py           # REWRITE: uses real dataloader
    run_hfold_inference.py           # keep (already plumbs aux paths)
    run_pipeline.py                  # NEW: orchestrates everything end-to-end
    # delete: probe_*, train_embedding_model.py, train_relevancy_model.py
  tests/
    test_heap.py                     # keep
    test_hfold_runtime.py            # keep + add chunked-eval invariants
    test_embedding_autoencoder.py    # keep
    test_relevancy_model.py          # extend with KL-target shape test
    test_model_hook_pythia.py        # keep
    test_model_hook_gpt2.py          # keep
    test_findings_regression.py      # keep
    test_extract_hidden_states.py    # NEW: smoke test that shapes match
    test_chunked_benchmark.py        # NEW: chunked HFold eval correctness
    test_aux_loading_e2e.py          # NEW: trained aux modules feed real folds
    test_training_smoke.py           # keep but point at real shards
    test_determinism.py              # keep
    test_fold_equation.py            # keep
    test_ablations.py                # keep
top-level:
  fine_tune.py                       # keep
  new_fine_tune.py                   # keep
  HFOLD_PIPELINE_PLAN.md             # this file
  README.md                          # rewrite the “run these commands” section
```

Files to be deleted (cleanup of debugging/probe artifacts):

- `hfold/_debug_log.py`
- `hfold/scripts/probe_findings.py`
- `hfold/scripts/probe_findings_v2.py`
- `hfold/scripts/probe_kv_cache.py`
- `hfold/scripts/probe_complexity.py`
- `hfold/scripts/train_embedding_model.py` (replaced by `train_aux_models.py`)
- `hfold/scripts/train_relevancy_model.py` (replaced by `train_aux_models.py`)
- `.cursor/debug-*.log` (any leftover debug logs)

---

## 7. User-Facing UX (the "few commands")

After approval and implementation, this is the entire developer flow. Each step is one command and idempotent.

### 7.1 Step 1 — Fine-tune the backbones (already exists)

```
python new_fine_tune.py   # configured for Pythia
# edit CONFIG.model_name = "gpt2", then:
python new_fine_tune.py
```

Outputs:
- `./checkpoints/pythia_full_finetuning/`
- `./checkpoints/gpt2_full_finetuning/`

(We will also add `python -m hfold.scripts.run_pipeline --stage finetune --backbone {pythia,gpt2}` as a convenience wrapper around `new_fine_tune.py`.)

### 7.2 Step 2 — Extract real hidden-state shards

```
python -m hfold.scripts.extract_hidden_states \
  --backbone pythia \
  --checkpoint-dir ./checkpoints/pythia_full_finetuning \
  --dataset wikitext \
  --output-dir ./data/extracted/pythia \
  --num-shards 8

python -m hfold.scripts.extract_hidden_states \
  --backbone gpt2 \
  --checkpoint-dir ./checkpoints/gpt2_full_finetuning \
  --dataset wikitext \
  --output-dir ./data/extracted/gpt2 \
  --num-shards 8
```

Outputs:
- `./data/extracted/pythia/shard_0000.pt … shard_0007.pt`
- `./data/extracted/gpt2/shard_0000.pt … shard_0007.pt`

Each shard is a list of dicts: `{ "backbone": str, "heap_vectors": [S, hidden], "evicted_vectors": [S, hidden], "teacher_scores": [S] }`.

### 7.3 Step 3 — Train aux models on real data

```
python -m hfold.scripts.train_aux_models \
  --extracted-dirs ./data/extracted/pythia ./data/extracted/gpt2 \
  --output-dir ./checkpoints/aux \
  --epochs 3
```

Outputs:
- `./checkpoints/aux/embedding_autoencoder.pt`
- `./checkpoints/aux/relevancy_transformer.pt`
- `./checkpoints/aux/adapters.pt`

### 7.4 Step 4 — Run all three benchmarks for both backbones

```
python -m hfold.scripts.benchmark_all_modes \
  --backbone pythia \
  --checkpoint-dir ./checkpoints/pythia_full_finetuning \
  --aux-dir ./checkpoints/aux \
  --dataset wikitext

python -m hfold.scripts.benchmark_all_modes \
  --backbone gpt2 \
  --checkpoint-dir ./checkpoints/gpt2_full_finetuning \
  --aux-dir ./checkpoints/aux \
  --dataset wikitext
```

Each prints one row per mode: `mode | loss | perplexity | tokens_per_second`.

### 7.5 Optional — Whole pipeline in one command

```
python -m hfold.scripts.run_pipeline \
  --backbones pythia gpt2 \
  --dataset wikitext \
  --output-root ./run_artifacts
```

Internally chains §7.1 → §7.4. Re-running with existing artifacts skips completed stages.

---

## 8. Test Plan (must pass before we say it works)

We add tests in three layers; existing test suite (currently 17 passing) is preserved.

### 8.1 Unit tests (existing + new)

- Heap, runtime, fold equation, mask expansion, top-w selection, dedup — all already covered.
- New: `test_extract_hidden_states.py`
  - mocks a tiny pretrained-shape model, runs the extractor on 1 batch, asserts shard schema and shapes.
- New: `test_relevancy_kl_targets.py`
  - asserts that teacher_scores are valid distributions (sum≈1, ≥0).

### 8.2 Integration tests (new)

- `test_chunked_benchmark.py`
  - Builds a tiny Pythia-shaped model, runs `_run_eval` over a 3-batch chunked dataloader, asserts:
    - runtime resets between sequences (already covered),
    - `call_count == num_chunks_per_sequence` within a sequence,
    - heap state contains entries from final chunk only when expected.
- `test_aux_loading_e2e.py`
  - Trains aux models for 5 steps on 16 fake shards, saves checkpoints, builds a Pythia-HFold bundle pointing at those checkpoints, runs one forward, asserts the heap update used the loaded weights (parameter equality check on a sentinel layer).

### 8.3 Smoke tests (new)

- `test_pipeline_smoke.py`
  - End-to-end test using `EleutherAI/pythia-31m` (the smallest real Pythia, ~31M params, fits in CPU memory): runs steps §7.2 → §7.4 with `--num-shards 1 --epochs 1 --max-eval-batches 2`. Asserts all artifacts exist and benchmark prints three rows.
  - Marked `@pytest.mark.slow` (skipped by default, runs in CI nightly).

### 8.4 Determinism

- `test_determinism.py`
  - With fixed seeds, the HFold benchmark loss is bitwise reproducible across two runs on the same machine.

### 8.5 Test gating

- Regular `pytest` runs all fast unit + integration tests (no model downloads).
- `pytest -m slow` runs the smoke pipeline test.
- Pre-merge gate: all fast tests + lints clean.

---

## 9. Correctness Checklist (Algorithm vs Code Mapping)

| Algorithm step | Code location | How we verify |
|---|---|---|
| timestep 0: forward, no retrieval, push top-w to heap | `model_hook.py: timestep == 0` branch | `test_global_hook_uses_single_heap_regardless_of_layer_count`, new `test_chunked_benchmark` |
| timestep t: pop K, prepend, forward | `model_hook.py: timestep > 0` + `vector_store.append_heap_vectors` | `test_global_hook_aligns_attention_mask_after_prepend` + new chunked test |
| reinsert K-transformed + top-w (de-duplicated) | `hfold_runtime.step_with_reinsert_and_fold` | `test_runtime_dedupes_popped_token_positions_from_top_w` |
| evicted set → embedding model → summary | `_fold_current_heap` calls `embedding_model.encode_summary` | `test_aux_loading_e2e` checks summary not None when evictions occur |
| relevancy.score_heap → r_i | same | `test_relevancy_model.py` shape & range checks |
| h_i ← h_i + r_i · g (fold) | `_fold_current_heap` | `test_fold_equation.py` (existing) |
| heap is global across all timesteps | single `GLOBAL_HEAP_INDEX = 0` | `test_global_hook_uses_single_heap_regardless_of_layer_count` |
| heap reset between sequences in eval | `_run_eval` calls `runtime.reset()` | `test_run_eval_resets_hfold_runtime_between_batches` |

---

## 10. Hard "no placeholders" constraints

- The synthetic `SyntheticHiddenStateDataset` and `_dummy_lm_collate` are removed. No code path may rely on them.
- Aux model construction in runners refuses to silently use random weights. If any of the three checkpoint paths (embedding/relevancy/adapters) is missing, the runner raises `RuntimeError` unless `--allow-random-aux` is explicitly passed.
- All training scripts save full state_dicts; all loading scripts use `weights_only=True` (silences the FutureWarning we currently emit).
- All scripts have `--help` and exit non-zero on misuse.

---

## 11. Risks and Tradeoffs

- **No KV cache during HFold inference**: confirmed unavoidable. Reflects in tokens/sec; we will report this honestly.
- **Aux models trained on extracted last-layer states only**: layer-MoE is deferred. We document this clearly in the report.
- **Extractor disk usage**: 8 shards × ~5MB ≈ 40MB per backbone. Acceptable.
- **WikiText-103 is the canonical eval dataset**: matches existing fine-tune. SCROLLS GovReport is optional and supported but not the default.
- **Full pipeline runtime on a laptop CPU**: extraction ~10 minutes per backbone, aux training ~5 minutes, benchmark ~5 minutes per backbone. Total ≈ 35 minutes for the whole flow on `pythia-31m` + `gpt2`.

---

## 12. Acceptance criteria (what "done" looks like)

1. The four commands in §7 run on a clean checkout (after `pip install -r requirements.txt`) and produce three benchmark rows for each backbone.
2. `pytest -q` passes with all new tests.
3. No file under `hfold/` references "synthetic" or "placeholder".
4. The three benchmark numbers per backbone are sensible: `loss(full_attention) ≤ loss(hfold) ≤ loss(sliding_window)` *or* HFold beats sliding-window (depending on training quality). Hard inequality is not required to call this "done", but the numbers must not be NaN/inf and tokens/sec must be positive for all three.
5. Determinism test passes.

---

## 13. What I need from you to proceed

Reply "go" (or with edits) and I will execute §6 in this order:

1. Delete cleanup files.
2. Add `data/extract_hidden_states.py` + tests.
3. Rewrite `hidden_state_dataset.py` to read shards.
4. Rewrite `train_embedding.py` and `train_relevancy.py` to use real shards and save state_dicts.
5. Add `scripts/extract_hidden_states.py`, `scripts/train_aux_models.py`, `scripts/run_pipeline.py`.
6. Rewrite `scripts/benchmark_all_modes.py` to use the real dataloader and chunked HFold.
7. Add chunked HFold support to `_run_eval` and `model_hook.py`.
8. Tighten runner defaults (no silent random aux models).
9. Add the new tests (§8) and remove debug helpers.
10. Update `README.md` with the §7 commands.

Each numbered step is one commit-sized change with green tests before moving to the next.
