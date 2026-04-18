from __future__ import annotations

import inspect
import logging
from types import MethodType
from typing import Any

import torch

from ..attention.hfold_backend import load_hfold_backend
from ..attention.registry import build_attention_strategy
from ..config import AttentionConfig, ExperimentConfig, normalize_attention_type

logger = logging.getLogger(__name__)


def resolve_torch_dtype(dtype_name: str) -> torch.dtype | None:
    lookup = {
        "auto": None,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype_name not in lookup:
        raise ValueError(f"Unsupported torch dtype '{dtype_name}'.")
    return lookup[dtype_name]


def _is_candidate_attention_module(module: torch.nn.Module) -> bool:
    has_qkv = hasattr(module, "query_key_value")
    has_out = hasattr(module, "dense")
    return has_qkv and has_out and callable(getattr(module, "forward", None))


def iter_pythia_attention_modules(model: torch.nn.Module):
    layer_index = 0
    for name, module in model.named_modules():
        if name.endswith("attention") and _is_candidate_attention_module(module):
            yield layer_index, name, module
            layer_index += 1


def patch_model_attention(
    model: torch.nn.Module,
    attention_config: AttentionConfig,
) -> torch.nn.Module:
    attention_type = normalize_attention_type(attention_config.attention_type)
    attention_config.attention_type = attention_type
    backend = None
    if attention_type == "hfold":
        backend = load_hfold_backend(attention_config.hfold_backend, attention_config.hfold)
    patched_layers = 0

    for layer_index, name, module in iter_pythia_attention_modules(model):
        patch_attention_module(
            module=module,
            layer_index=layer_index,
            attention_config=attention_config,
            hfold_backend=backend,
        )
        logger.info(
            "Patched attention layer %s with mode=%s.",
            name,
            module._hfold_attention_type,
        )
        patched_layers += 1

    if patched_layers == 0:
        raise RuntimeError("No GPT-NeoX attention modules were found to patch.")

    return model


def patch_attention_module(
    *,
    module: torch.nn.Module,
    layer_index: int,
    attention_config: AttentionConfig | None = None,
    hfold_backend: Any | None = None,
    strategy: Any | None = None,
) -> None:
    if strategy is None:
        if attention_config is None:
            raise TypeError(
                "patch_attention_module requires either a ready-made strategy or "
                "attention_config/hfold_backend."
            )
        strategy = build_attention_strategy(attention_config, hfold_backend=hfold_backend)
    strategy.prepare_module(module=module, layer_index=layer_index)

    original_forward = getattr(module, "_hfold_original_forward", module.forward)
    signature = inspect.signature(original_forward)

    def _patched_forward(_self: torch.nn.Module, *args: Any, **kwargs: Any) -> Any:
        bound = signature.bind_partial(*args, **kwargs)
        if "hidden_states" not in bound.arguments and args:
            bound.arguments["hidden_states"] = args[0]
        return strategy.invoke(
            module=module,
            original_forward=original_forward,
            bound_arguments=bound.arguments,
            layer_index=layer_index,
        )

    module._hfold_original_forward = original_forward
    module._hfold_attention_type = strategy.name
    module._hfold_layer_index = layer_index
    module.forward = MethodType(_patched_forward, module)


def load_pythia_tokenizer(experiment_config: ExperimentConfig):
    from transformers import AutoTokenizer

    model_cfg = experiment_config.model
    tokenizer_name = model_cfg.tokenizer_name or model_cfg.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir=model_cfg.cache_dir,
        trust_remote_code=model_cfg.trust_remote_code,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_pythia_model_and_tokenizer(experiment_config: ExperimentConfig):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_cfg = experiment_config.model
    dtype = resolve_torch_dtype(model_cfg.torch_dtype)

    hf_config = AutoConfig.from_pretrained(
        model_cfg.model_name,
        cache_dir=model_cfg.cache_dir,
        trust_remote_code=model_cfg.trust_remote_code,
    )
    if hasattr(hf_config, "_attn_implementation"):
        hf_config._attn_implementation = model_cfg.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_name,
        config=hf_config,
        cache_dir=model_cfg.cache_dir,
        trust_remote_code=model_cfg.trust_remote_code,
        torch_dtype=dtype,
    )
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = model_cfg.attn_implementation

    tokenizer = load_pythia_tokenizer(experiment_config)

    patch_model_attention(model, experiment_config.attention)

    if model_cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if model_cfg.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    return model, tokenizer
