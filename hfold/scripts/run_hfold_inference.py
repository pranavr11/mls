from __future__ import annotations

import argparse

import torch

from hfold.config.schema import HFoldConfig, HFoldModelConfig, HFoldTrainingConfig
from hfold.integration.gpt2_runner import build_gpt2_with_hfold
from hfold.integration.pythia_runner import build_pythia_with_hfold


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["pythia", "gpt2"], required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prompt", default="Hello world")
    parser.add_argument("--embedding-checkpoint", default=None)
    parser.add_argument("--relevancy-checkpoint", default=None)
    parser.add_argument("--adapters-checkpoint", default=None)
    parser.add_argument("--max-heap-size", type=int, default=16)
    parser.add_argument("--adapter-dim", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    # hidden_size/num_heads are placeholders — the runner overrides them from
    # the loaded model's actual config.
    config = HFoldConfig(
        model=HFoldModelConfig(
            hidden_size=1,
            num_heads=1,
            max_heap_size=args.max_heap_size,
            top_w=args.max_heap_size,
            pop_k=args.max_heap_size,
            adapter_dim=args.adapter_dim,
        ),
        training=HFoldTrainingConfig(),
    )
    aux_kwargs = dict(
        embedding_checkpoint_path=args.embedding_checkpoint,
        relevancy_checkpoint_path=args.relevancy_checkpoint,
        adapters_checkpoint_path=args.adapters_checkpoint,
    )
    if args.backbone == "pythia":
        bundle = build_pythia_with_hfold(
            model_name=args.model_name,
            checkpoint_path=args.checkpoint,
            config=config,
            **aux_kwargs,
        )
    else:
        bundle = build_gpt2_with_hfold(
            model_name=args.model_name,
            checkpoint_path=args.checkpoint,
            config=config,
            **aux_kwargs,
        )
    tokenizer = bundle.tokenizer
    model = bundle.model
    inputs = tokenizer(args.prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, use_cache=False)
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
