# Pythia Attention Surface

This note fixes the minimal modification surface for the project.

## Backbone choice

- checkpoint: `EleutherAI/pythia-160m`
- architecture family: `GPT-NeoX`
- task form: decoder-only causal language modeling

For this project, we treat Pythia-160M as a Hugging Face `GPTNeoXForCausalLM`-style model whose attention path is the only experimental variable.

## What must stay unchanged

To preserve pretrained initialization and make the ablation fair, the following should remain untouched across runs:

- token embeddings
- positional encoding scheme already used by GPT-NeoX/Pythia
- transformer block structure
- layer norms
- MLP weights and activations
- residual connections
- QKV projection weights
- attention output projection weights
- tokenizer
- optimizer and scheduler recipe
- PG-19 preprocessing contract

## What we intercept

The intervention point is the GPT-NeoX attention module inside each transformer layer.

At a high level:

```text
GPTNeoXForCausalLM
  -> GPTNeoXModel
    -> layers[i]
      -> attention
```

Inside each attention module, the relevant pretrained components are:

- `query_key_value`
- `dense`

Those weights are reused exactly as loaded from the pretrained checkpoint.

## Current implementation strategy in this repo

We patch the attention module `forward` method in place instead of replacing the entire module. That choice keeps checkpoint/state-dict keys stable while still allowing the attention behavior to change.

The patch logic lives in:

- [src/hfold_pipeline/modeling/pythia.py](/Users/krishmody/Fold2/src/hfold_pipeline/modeling/pythia.py)

The attention strategies live in:

- [src/hfold_pipeline/attention/full.py](/Users/krishmody/Fold2/src/hfold_pipeline/attention/full.py)
- [src/hfold_pipeline/attention/sliding_window.py](/Users/krishmody/Fold2/src/hfold_pipeline/attention/sliding_window.py)
- [src/hfold_pipeline/attention/hfold.py](/Users/krishmody/Fold2/src/hfold_pipeline/attention/hfold.py)

## Exact swap boundary

The only thing that changes between `full`, `sliding_window`, and future `hfold` is:

- the attention mask/pattern
- any extra HFold memory logic injected through the HFold backend hook

## Current HFold status

HFold is now implemented in-tree and is attached through the same backend contract that the original scaffold reserved for it. The merged backend keeps the training pipeline unchanged while registering the HFold-specific fold/memory parameters on each patched GPT-NeoX attention layer.

Everything else in the forward pass should be preserved as inherited from GPT-NeoX.

## Why this is the right insertion point

- It preserves compatibility with pretrained Pythia weights.
- It isolates the experimental variable to attention.
- It lets HFold remain a drop-in backend instead of a model-wide fork.
- It supports apples-to-apples comparisons across attention variants.
