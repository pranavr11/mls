from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config.schema import HFoldConfig
from ..inference.attention_patch import patch_gpt2_model_attention
from ..inference.hfold_runtime import HFoldRuntime
from ..models.adapters import BackboneAdapterRegistry
from ..models.embedding_autoencoder import EmbeddingAutoencoder
from ..models.relevancy_transformer import RelevancyTransformer


@dataclass
class HFoldGPT2Bundle:
    model: torch.nn.Module
    tokenizer: AutoTokenizer
    runtime: HFoldRuntime
    embedding_model: EmbeddingAutoencoder
    relevancy_model: RelevancyTransformer


def build_gpt2_with_hfold(
    *,
    model_name: str,
    checkpoint_path: str | None,
    config: HFoldConfig,
    cache_dir: str = "./data",
) -> HFoldGPT2Bundle:
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if checkpoint_path:
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, cache_dir=cache_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
    runtime = HFoldRuntime(config)
    embedding_model = EmbeddingAutoencoder(
        hidden_size=config.model.adapter_dim,
        latent_size=config.model.adapter_dim,
        max_slots=config.model.max_heap_size,
    )
    relevancy_model = RelevancyTransformer(hidden_size=config.model.adapter_dim)
    adapters = BackboneAdapterRegistry(specs={"pythia": config.model.hidden_size, "gpt2": config.model.hidden_size}, shared_dim=config.model.adapter_dim)
    runtime.attach_adapters(adapters, "gpt2")
    patch_gpt2_model_attention(model, runtime, embedding_model, relevancy_model)
    return HFoldGPT2Bundle(
        model=model,
        tokenizer=tokenizer,
        runtime=runtime,
        embedding_model=embedding_model,
        relevancy_model=relevancy_model,
    )
