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
    return parser.parse_args()


def main():
    args = parse_args()
    config = HFoldConfig(model=HFoldModelConfig(hidden_size=768, num_heads=12), training=HFoldTrainingConfig())
    if args.backbone == "pythia":
        bundle = build_pythia_with_hfold(model_name=args.model_name, checkpoint_path=args.checkpoint, config=config)
    else:
        bundle = build_gpt2_with_hfold(model_name=args.model_name, checkpoint_path=args.checkpoint, config=config)
    tokenizer = bundle.tokenizer
    model = bundle.model
    inputs = tokenizer(args.prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=20)
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
