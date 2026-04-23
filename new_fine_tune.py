"""
This file contains all the necessary information for running fine-tuning of small transformer
models with full attention on long-context tasks. The initial model will use pretrained weights from 
GPT-2 and Pythia-31M, running on the WikiText (wikipedia articles) and SCROLLS (a set of 
long-context reasoning benchmarks) datasets. 
"""

from dotenv import load_dotenv
import os
import math
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AdamW,
    get_cosine_schedule_with_warmup,
    DataCollatorForLanguageModeling,
)
from tqdm import tqdm
import random

load_dotenv()

# NOTING: THIS DOES NOT WORK FOR SCROLLS YET

CONFIG = {
    "cache_dir": "./data",
    "output_dir": "./checkpoints",

    "model_name": "EleutherAI/pythia-31m",   # or "gpt2"
    "dataset_name": "wikitext",              # "wikitext" or "scrolls_gov_report"
    "wikitext_config": "wikitext-103-raw-v1",

    "max_length": 2048,
    "train_batch_size": 4,
    "eval_batch_size": 4,

    "phase1_epochs": 2,          # k
    "phase2_epochs": 3,          # m
    "last_k_layers": 2,          # start with 2; change to 1 if needed

    "lr_phase1": 5e-5,
    "lr_phase2": 1e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,

    "save_phase1_name": "last_k_finetuning",
    "save_phase2_name": "full_finetuning",

    "attention-format" : "regular",
    "seed" : 42,

    "sliding-length" : 256
}


device = "cuda" if torch.cuda.is_available() else "cpu"



def load_raw_dataset(config):
    if config["dataset_name"] == "wikitext":
        dataset = load_dataset(
            "wikitext",
            config["wikitext_config"],
            cache_dir=config["cache_dir"],
        )
    elif config["dataset_name"] == "scrolls_gov_report":
        dataset = load_dataset(
            "tau/scrolls",
            config["scrolls_config"],
            cache_dir=config["cache_dir"],
            trust_remote_code=True,
        )
    else:
        raise ValueError(f"Unsupported dataset_name: {config['dataset_name']}")
    return dataset


def tokenize_function(examples, tokenizer, config):
    if config["dataset_name"] == "wikitext":
        texts = examples["text"]
    elif config["dataset_name"] == "scrolls_gov_report":
        texts = [
            f"Document:\n{inp}\n\nSummary:\n{out}"
            for inp, out in zip(examples["input"], examples["output"])
        ]
    else:
        raise ValueError(f"Unsupported dataset_name: {config['dataset_name']}")

    return tokenizer(
        texts,
        truncation=True,
        max_length=config["max_length"],
        padding=False,
    )


def group_texts(examples, block_size):
    concatenated = {}
    for k in examples.keys():
        concatenated[k] = sum(examples[k], [])

    total_length = len(concatenated["input_ids"])
    total_length = (total_length // block_size) * block_size

    result = {}
    for k, v in concatenated.items():
        result[k] = [v[i:i + block_size] for i in range(0, total_length, block_size)]

    result["labels"] = result["input_ids"].copy()
    return result


def build_dataloaders(tokenizer, config):
    raw_dataset = load_raw_dataset(config)

    if config["dataset_name"] == "wikitext":
        tokenized = raw_dataset.map(
            lambda examples: tokenizer(examples["text"], add_special_tokens=False),
            batched=True,
            remove_columns=raw_dataset["train"].column_names,
        )

        lm_dataset = tokenized.map(
            lambda examples: group_texts(examples, config["max_length"]),
            batched=True,
        )

        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        train_loader = DataLoader(
            lm_dataset["train"],
            batch_size=config["train_batch_size"],
            shuffle=True,
            collate_fn=collator,
        )

        eval_split = "validation" if "validation" in lm_dataset else "test"
        eval_loader = DataLoader(
            lm_dataset[eval_split],
            batch_size=config["eval_batch_size"],
            shuffle=False,
            collate_fn=collator,
        )

    elif config["dataset_name"] == "scrolls_gov_report":
        tokenized = raw_dataset.map(
            lambda examples: tokenize_function(examples, tokenizer, config),
            batched=True,
            remove_columns=raw_dataset["train"].column_names,
        )

        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        train_loader = DataLoader(
            tokenized["train"],
            batch_size=config["train_batch_size"],
            shuffle=True,
            collate_fn=collator,
        )

        eval_split = "validation" if "validation" in tokenized else "test"
        eval_loader = DataLoader(
            tokenized[eval_split],
            batch_size=config["eval_batch_size"],
            shuffle=False,
            collate_fn=collator,
        )

    else:
        raise ValueError(f"Unsupported dataset_name: {config['dataset_name']}")

    return train_loader, eval_loader


def create_optimizer(model, lr, weight_decay):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return AdamW(trainable_params, lr=lr, weight_decay=weight_decay)


def create_scheduler(optimizer, dataloader, num_epochs, warmup_ratio):
    total_steps = len(dataloader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return scheduler


def reinitialize_last_k_layers(model, k):
    modules_to_reset = []

    # Pythia / GPT-NeoX
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        modules_to_reset.extend(model.gpt_neox.layers[-k:])
        
        # add final layer norm and LM head
        if hasattr(model.gpt_neox, "final_layer_norm"):
            modules_to_reset.append(model.gpt_neox.final_layer_norm)
        if hasattr(model, "embed_out"):
            modules_to_reset.append(model.embed_out)

    # GPT-2
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        modules_to_reset.extend(model.transformer.h[-k:])
        
        # add final layer norm and LM head
        if hasattr(model.transformer, "ln_f"):
            modules_to_reset.append(model.transformer.ln_f)
        if hasattr(model, "lm_head"):
            modules_to_reset.append(model.lm_head)

    else:
        raise ValueError("Unsupported model architecture for reinitialization.")

    for target_module in modules_to_reset:
        for module in target_module.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()


def freeze_all_but_last_k_layers(model, k):
    for param in model.parameters():
        param.requires_grad = False

    # Pythia / GPT-NeoX
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        layers = model.gpt_neox.layers
        for layer in layers[-k:]:
            for param in layer.parameters():
                param.requires_grad = True

        if hasattr(model.gpt_neox, "final_layer_norm"):
            for param in model.gpt_neox.final_layer_norm.parameters():
                param.requires_grad = True

        if hasattr(model, "embed_out"):
            for param in model.embed_out.parameters():
                param.requires_grad = True

    # GPT-2
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
        for layer in layers[-k:]:
            for param in layer.parameters():
                param.requires_grad = True

        if hasattr(model.transformer, "ln_f"):
            for param in model.transformer.ln_f.parameters():
                param.requires_grad = True

        if hasattr(model, "lm_head"):
            for param in model.lm_head.parameters():
                param.requires_grad = True

    else:
        raise ValueError("Unsupported model architecture for freezing.")


class SlidingWindowAttentionWrapper(torch.nn.Module):
    """
    Wraps an existing HF attention module to enforce a sliding-window constraint
    on the attention mask during the forward pass.

    Handles asymmetric (non-square) masks so it works correctly during both
    training and auto-regressive generation with KV caching. Supports both
    additive float masks and boolean masks.
    """
    def __init__(self, original_attention, window_size):
        super().__init__()
        self.original_attention = original_attention
        self.window_size = window_size

    def forward(self, hidden_states, *args, **kwargs):
        if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
            mask = kwargs["attention_mask"]
            device = mask.device

            # Support non-square masks (tgt_len=1 during KV-cache generation)
            tgt_len = mask.shape[-2]
            src_len = mask.shape[-1]

            idx_tgt = torch.arange(src_len - tgt_len, src_len, device=device).unsqueeze(1)
            idx_src = torch.arange(src_len, device=device).unsqueeze(0)

            out_of_window = (idx_tgt - idx_src) >= self.window_size

            modified_mask = mask.clone()
            if modified_mask.dtype == torch.bool:
                modified_mask = modified_mask.masked_fill(out_of_window, False)
            else:
                min_val = torch.finfo(modified_mask.dtype).min
                modified_mask = modified_mask.masked_fill(out_of_window, min_val)

            kwargs["attention_mask"] = modified_mask

        return self.original_attention(hidden_states, *args, **kwargs)


def apply_sliding_window_to_last_k_layers(model, k, window_size):
    """
    Replaces the attention module in the last k transformer layers with a
    SlidingWindowAttentionWrapper. Earlier layers retain full attention.
    Supports Pythia/GPT-NeoX and GPT-2.
    """
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        for layer in model.gpt_neox.layers[-k:]:
            layer.attention = SlidingWindowAttentionWrapper(layer.attention, window_size)
            print(f"Applied sliding-window attention (window={window_size}) to Pythia layer.")

    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        for layer in model.transformer.h[-k:]:
            layer.attn = SlidingWindowAttentionWrapper(layer.attn, window_size)
            print(f"Applied sliding-window attention (window={window_size}) to GPT-2 layer.")

    else:
        raise ValueError("Unsupported model architecture for sliding window.")


def unfreeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = True


def save_checkpoint(model, tokenizer, output_dir, checkpoint_name):
    save_path = os.path.join(output_dir, checkpoint_name)
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Saved checkpoint to {save_path}")


def train_one_epoch(model, dataloader, optimizer, scheduler):
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        total_loss += outputs.loss.item()

    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return avg_loss, perplexity


def run_last_k_finetuning(model, tokenizer, train_loader, eval_loader, config):
    print(f"\nStarting destructive last-{config['last_k_layers']}-layer fine-tuning...")

    # destroy the last k layers (+ LN and LM Head)
    reinitialize_last_k_layers(model, config["last_k_layers"])

    # freeze everything else
    freeze_all_but_last_k_layers(model, config["last_k_layers"])

    optimizer = create_optimizer(
        model,
        config["lr_phase1"],
        config["weight_decay"],
    )
    scheduler = create_scheduler(
        optimizer,
        train_loader,
        config["phase1_epochs"],
        config["warmup_ratio"],
    )

    for epoch in range(config["phase1_epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler)
        eval_loss, eval_ppl = evaluate(model, eval_loader)

        print(
            f"[Phase 1 | Epoch {epoch + 1}/{config['phase1_epochs']}] "
            f"train_loss={train_loss:.4f} | eval_loss={eval_loss:.4f} | eval_ppl={eval_ppl:.4f}"
        )

    save_checkpoint(model, tokenizer, config["output_dir"], config["save_phase1_name"])


def run_full_finetuning(model, tokenizer, train_loader, eval_loader, config):
    print("\nStarting full fine-tuning...")

    # keep current weights, just unfreeze all layers
    unfreeze_all_layers(model)

    # create a new optimizer for Phase 2 now that all layers are unfrozen
    optimizer = create_optimizer(
        model,
        config["lr_phase2"],
        config["weight_decay"],
    )

    scheduler = create_scheduler(
        optimizer,
        train_loader,
        config["phase2_epochs"],
        config["warmup_ratio"],
    )

    for epoch in range(config["phase2_epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler)
        eval_loss, eval_ppl = evaluate(model, eval_loader)

        print(
            f"[Phase 2 | Epoch {epoch + 1}/{config['phase2_epochs']}] "
            f"train_loss={train_loss:.4f} | eval_loss={eval_loss:.4f} | eval_ppl={eval_ppl:.4f}"
        )

    save_checkpoint(model, tokenizer, config["output_dir"], config["save_phase2_name"])


def main():
    seed = CONFIG['seed']
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    print("Loading model/tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["model_name"],
        cache_dir=CONFIG["cache_dir"],
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        cache_dir=CONFIG["cache_dir"],
    ).to(device)

    # Apply sliding-window attention to the last k layers before any training
    if CONFIG["attention-format"] == "sliding-window":
        apply_sliding_window_to_last_k_layers(
            model, CONFIG["last_k_layers"], CONFIG["sliding-length"]
        )

    print("Building dataloaders...")
    train_loader, eval_loader = build_dataloaders(tokenizer, CONFIG)

    run_last_k_finetuning(model, tokenizer, train_loader, eval_loader, CONFIG)
    run_full_finetuning(model, tokenizer, train_loader, eval_loader, CONFIG)



if __name__ == "__main__":
    main()