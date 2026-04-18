from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_name: str, cache_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
    model.config.output_attentions = True
    return model, tokenizer
