from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config.schema import HFoldConfig
from ..inference.hfold_runtime import HFoldRuntime
from ..inference.model_hook import wrap_pythia_with_hfold
from ..models.adapters import BackboneAdapterRegistry
from ..models.embedding_autoencoder import EmbeddingAutoencoder
from ..models.relevancy_transformer import RelevancyTransformer


@dataclass
class HFoldPythiaBundle:
    model: torch.nn.Module
    tokenizer: AutoTokenizer
    runtime: HFoldRuntime
    embedding_model: EmbeddingAutoencoder
    relevancy_model: RelevancyTransformer


def build_pythia_with_hfold(
    *,
    model_name: str,
    checkpoint_path: str | None,
    config: HFoldConfig,
    cache_dir: str = "./data",
    embedding_checkpoint_path: str | None = None,
    relevancy_checkpoint_path: str | None = None,
    adapters_checkpoint_path: str | None = None,
    backbone_dims: dict[str, int] | None = None,
) -> HFoldPythiaBundle:
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if checkpoint_path:
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, cache_dir=cache_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
    detected_hidden = int(model.config.hidden_size)
    detected_heads = int(getattr(model.config, "num_attention_heads", config.model.num_heads))
    config.model.hidden_size = detected_hidden
    config.model.num_heads = detected_heads
    config.model.validate()
    runtime = HFoldRuntime(config)
    embedding_model = EmbeddingAutoencoder(
        hidden_size=config.model.adapter_dim,
        latent_size=int(config.model.embedding_latent_dim),
        max_slots=config.model.max_heap_size,
    )
    relevancy_model = RelevancyTransformer(hidden_size=config.model.adapter_dim)
    specs = dict(backbone_dims) if backbone_dims else {"pythia": detected_hidden}
    specs["pythia"] = detected_hidden
    adapters = BackboneAdapterRegistry(
        specs=specs,
        shared_dim=config.model.adapter_dim,
    )
    if adapters_checkpoint_path:
        adapters.load_state_dict(
            torch.load(adapters_checkpoint_path, map_location="cpu", weights_only=True),
            strict=False,
        )
    if embedding_checkpoint_path:
        embedding_model.load_state_dict(
            torch.load(embedding_checkpoint_path, map_location="cpu", weights_only=True),
        )
    if relevancy_checkpoint_path:
        relevancy_model.load_state_dict(
            torch.load(relevancy_checkpoint_path, map_location="cpu", weights_only=True),
        )
    runtime.attach_adapters(adapters, "pythia")
    wrap_pythia_with_hfold(model, runtime, embedding_model, relevancy_model)
    return HFoldPythiaBundle(
        model=model,
        tokenizer=tokenizer,
        runtime=runtime,
        embedding_model=embedding_model,
        relevancy_model=relevancy_model,
    )
