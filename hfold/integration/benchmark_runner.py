from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from ..config.schema import HFoldConfig
from .checkpoint_utils import load_gpt_neox_causal_lm_from_folder
from .gpt2_runner import build_gpt2_with_hfold
from .pythia_runner import build_pythia_with_hfold


def _move_hfold_bundle_to_device(model: torch.nn.Module, device_obj: torch.device) -> None:
    """Move the causal LM plus HFold submodules (embedding, relevancy, adapters)."""
    model.to(device_obj)
    # Legacy builds: adapters only on HFoldRuntime, not registered via add_module("hfold_adapters").
    runtime = getattr(model, "hfold_runtime", None)
    if runtime is not None:
        adapters = getattr(runtime, "_adapters", None)
        reg = getattr(model, "hfold_adapters", None)
        if adapters is not None and reg is not adapters:
            adapters.to(device_obj)


@dataclass
class BenchmarkResult:
    mode: str
    loss: float
    perplexity: float
    tokens_per_second: float


@torch.no_grad()
def _run_eval(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    *,
    hfold_window_size: int | None = None,
    hfold_eval_use_kv_cache: bool = True,
    hfold_eval_chunk_size: int = 1,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_batches = 0
    start = time.perf_counter()
    runtime = getattr(model, "hfold_runtime", None)
    for batch in dataloader:
        # Each evaluation example is an independent sequence; HFold heap state
        # must not leak across unrelated batches/sequences.
        if runtime is not None:
            runtime.reset()
        batch = {k: v.to(device) for k, v in batch.items()}
        if runtime is not None and "labels" in batch:
            # HFold semantics are autoregressive and timestep-based. For runtime
            # eval we score next-token NLL one step at a time so heap evolution
            # actually executes across timesteps.
            #
            # Fast path: if the model supports KV cache, feed one token per step
            # with `past_key_values` to avoid repeatedly re-encoding prefixes.
            # Fallback path: for tiny test stubs that do not accept cache kwargs,
            # preserve the legacy bounded-prefix behavior.
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask")
            batch_nll = 0.0
            batch_pred_tokens = 0
            for row in range(input_ids.size(0)):
                runtime.reset()
                row_ids = input_ids[row : row + 1]
                row_mask = None if attention_mask is None else attention_mask[row : row + 1]
                seq_len = int(row_ids.size(1))
                past_kv = None
                supports_cache = bool(hfold_eval_use_kv_cache)
                chunk_size = max(1, int(hfold_eval_chunk_size))
                next_pos = 1
                while next_pos < seq_len:
                    out = None
                    block_pred_tokens = min(chunk_size, seq_len - next_pos)
                    if supports_cache:
                        # Predict next `block_pred_tokens` from the block of
                        # preceding inputs and cached context.
                        input_start = next_pos - 1
                        input_end = input_start + block_pred_tokens
                        step_ids = row_ids[:, input_start:input_end]
                        step_mask = None
                        if row_mask is not None:
                            # Mask covers cached prefix + current block inputs.
                            context_end = input_end
                            step_mask = row_mask[:, :context_end].clone()
                            if hfold_window_size is not None and hfold_window_size > 0:
                                cutoff = max(0, context_end - int(hfold_window_size))
                                if cutoff > 0:
                                    step_mask[:, :cutoff] = 0
                        call_kwargs = {
                            "input_ids": step_ids,
                            "attention_mask": step_mask,
                            "use_cache": True,
                        }
                        if past_kv is not None:
                            call_kwargs["past_key_values"] = past_kv
                        try:
                            out = model(**call_kwargs)
                            if hasattr(out, "past_key_values"):
                                past_kv = out.past_key_values
                            elif hasattr(out, "past_key_value"):
                                past_kv = out.past_key_value
                            else:
                                past_kv = None
                        except TypeError:
                            # Minimal unit-test doubles may not accept
                            # use_cache/past_key_values. Fall back to legacy
                            # bounded-prefix execution.
                            supports_cache = False
                            past_kv = None

                    if out is None:
                        # Legacy fallback path: score one token at a time from
                        # an explicit prefix if cache kwargs are unsupported.
                        block_pred_tokens = 1
                        start_idx = 0
                        if hfold_window_size is not None and hfold_window_size > 0:
                            start_idx = max(0, next_pos - int(hfold_window_size))
                        prefix_ids = row_ids[:, start_idx:next_pos]
                        prefix_mask = None if row_mask is None else row_mask[:, start_idx:next_pos]
                        out = model(
                            input_ids=prefix_ids,
                            attention_mask=prefix_mask,
                        )
                    if not hasattr(out, "logits"):
                        # Fallback for tiny unit-test stubs that only emit loss.
                        batch_nll += float(out.loss.item())
                        batch_pred_tokens += 1
                        break
                    logits = out.logits[:, -block_pred_tokens:, :]
                    target = row_ids[:, next_pos : next_pos + block_pred_tokens]
                    token_nll = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        target.reshape(-1),
                        reduction="sum",
                    )
                    batch_nll += float(token_nll.item())
                    batch_pred_tokens += int(block_pred_tokens)
                    next_pos += int(block_pred_tokens)
            if batch_pred_tokens > 0:
                total_loss += batch_nll / batch_pred_tokens
                total_batches += 1
                total_tokens += int(batch_pred_tokens)
        else:
            output = model(**batch)
            total_loss += float(output.loss.item())
            total_batches += 1
            total_tokens += int(batch["input_ids"].numel())
    elapsed = max(time.perf_counter() - start, 1e-6)
    avg_loss = total_loss / max(total_batches, 1)
    tok_s = total_tokens / elapsed
    return avg_loss, tok_s


def _to_result(mode: str, loss: float, tok_s: float) -> BenchmarkResult:
    return BenchmarkResult(
        mode=mode,
        loss=loss,
        perplexity=float(math.exp(loss)) if loss < 50.0 else float("inf"),
        tokens_per_second=tok_s,
    )


class _SlidingWindowMaskWrapper(torch.nn.Module):
    """Inline sliding-window mask wrapper, mirroring `new_fine_tune.py` semantics."""

    def __init__(self, original_attention: torch.nn.Module, window_size: int) -> None:
        super().__init__()
        self.original_attention = original_attention
        self.window_size = window_size

    def forward(self, hidden_states, *args, **kwargs):
        if self.window_size <= 0:
            return self.original_attention(hidden_states, *args, **kwargs)

        def _window_block(src_len: int, tgt_len: int, device: torch.device) -> torch.Tensor:
            idx_tgt = torch.arange(src_len - tgt_len, src_len, device=device).unsqueeze(1)
            idx_src = torch.arange(src_len, device=device).unsqueeze(0)
            out_of_window = (idx_tgt - idx_src) >= self.window_size
            future = idx_src > idx_tgt
            return out_of_window | future

        def _make_additive_mask(blocked: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
            out = torch.zeros(blocked.shape, dtype=dtype, device=blocked.device)
            min_val = torch.finfo(dtype).min
            return out.masked_fill(blocked, min_val)

        # GPT-NeoX attention often receives `attention_mask` positionally as arg[0].
        # If this is None, we synthesize an explicit local causal mask so sliding-window
        # has effect even on SDPA-style call paths.
        is_neox = "NeoX" in type(self.original_attention).__name__
        mask = kwargs.get("attention_mask")
        mask_pos = None
        if mask is None and is_neox and args:
            candidate = args[0]
            if candidate is None or (torch.is_tensor(candidate) and candidate.dim() in (2, 4)):
                mask = candidate
                mask_pos = 0
        if mask is None and args:
            for i, candidate in enumerate(args):
                if torch.is_tensor(candidate) and candidate.dim() == 4:
                    mask = candidate
                    mask_pos = i
                    break

        modified = None
        if torch.is_tensor(mask) and mask.dim() == 4:
            tgt_len = mask.shape[-2]
            src_len = mask.shape[-1]
            blocked = _window_block(src_len, tgt_len, mask.device)
            modified = mask.clone()
            if modified.dtype == torch.bool:
                modified = modified.masked_fill(blocked, False)
            else:
                min_val = torch.finfo(modified.dtype).min
                modified = modified.masked_fill(blocked, min_val)
        elif torch.is_tensor(mask) and mask.dim() == 2:
            # Convert a 2D padding mask [B, S] into additive 4D [B, 1, T, S] and
            # apply both causal + local-window masking.
            batch, src_len = mask.shape
            tgt_len = hidden_states.shape[1]
            blocked = _window_block(src_len, tgt_len, hidden_states.device)
            blocked = blocked.unsqueeze(0).unsqueeze(0).expand(batch, 1, tgt_len, src_len)
            allowed_src = mask.to(torch.bool).unsqueeze(1).unsqueeze(1).expand(batch, 1, tgt_len, src_len)
            blocked = blocked | (~allowed_src)
            modified = _make_additive_mask(blocked, dtype=hidden_states.dtype)
        elif mask is None:
            # No mask provided to attention: enforce local causal mask directly.
            batch = hidden_states.shape[0]
            tgt_len = hidden_states.shape[1]
            src_len = tgt_len
            blocked = _window_block(src_len, tgt_len, hidden_states.device)
            blocked = blocked.unsqueeze(0).unsqueeze(0).expand(batch, 1, tgt_len, src_len)
            modified = _make_additive_mask(blocked, dtype=hidden_states.dtype)

        if modified is not None:
            if mask_pos is not None:
                args = list(args)
                args[mask_pos] = modified
                args = tuple(args)
            else:
                kwargs["attention_mask"] = modified
        return self.original_attention(hidden_states, *args, **kwargs)


def _apply_sliding_window(model: torch.nn.Module, window_size: int) -> torch.nn.Module:
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        for layer in model.gpt_neox.layers:
            layer.attention = _SlidingWindowMaskWrapper(layer.attention, window_size)
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        for layer in model.transformer.h:
            layer.attn = _SlidingWindowMaskWrapper(layer.attn, window_size)
    else:
        raise ValueError("Unsupported model architecture for sliding-window benchmark.")
    return model


def _build_hfold_model(
    *,
    backbone: str,
    model_name: str,
    checkpoint_path: str | None,
    config: HFoldConfig,
    embedding_checkpoint_path: str | None = None,
    relevancy_checkpoint_path: str | None = None,
    adapters_checkpoint_path: str | None = None,
) -> torch.nn.Module:
    aux_kwargs = dict(
        embedding_checkpoint_path=embedding_checkpoint_path,
        relevancy_checkpoint_path=relevancy_checkpoint_path,
        adapters_checkpoint_path=adapters_checkpoint_path,
    )
    if backbone == "pythia":
        return build_pythia_with_hfold(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            config=config,
            **aux_kwargs,
        ).model
    if backbone == "gpt2":
        return build_gpt2_with_hfold(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            config=config,
            **aux_kwargs,
        ).model
    raise ValueError("backbone must be one of: pythia, gpt2")


def eval_hfold_only(
    *,
    backbone: str,
    model_name: str,
    checkpoint_path: str | None,
    dataloader,
    config: HFoldConfig,
    device: str = "cpu",
    sliding_window_size: int = 256,
    hfold_eval_use_kv_cache: bool = True,
    hfold_eval_chunk_size: int = 1,
    embedding_checkpoint_path: str | None = None,
    relevancy_checkpoint_path: str | None = None,
    adapters_checkpoint_path: str | None = None,
    mode_label: str = "hfold",
) -> BenchmarkResult:
    """
    Run perplexity eval with HFold hooked inference only (one model load).

    Use this to compare HFold PPL across different checkpoints without paying
    for full- and sliding-window baselines in the same process.
    """
    device_obj = torch.device(device)
    model = _build_hfold_model(
        backbone=backbone,
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        config=config,
        embedding_checkpoint_path=embedding_checkpoint_path,
        relevancy_checkpoint_path=relevancy_checkpoint_path,
        adapters_checkpoint_path=adapters_checkpoint_path,
    )
    _move_hfold_bundle_to_device(model, device_obj)
    loss, tok_s = _run_eval(
        model,
        dataloader,
        device_obj,
        hfold_window_size=sliding_window_size,
        hfold_eval_use_kv_cache=hfold_eval_use_kv_cache,
        hfold_eval_chunk_size=hfold_eval_chunk_size,
    )
    return _to_result(mode_label, loss, tok_s)


def benchmark_three_modes(
    *,
    backbone: str,
    model_name: str,
    checkpoint_path: str | None,
    dataloader,
    config: HFoldConfig,
    device: str = "cpu",
    sliding_window_size: int = 256,
    hfold_eval_use_kv_cache: bool = True,
    hfold_eval_chunk_size: int = 1,
    full_model_factory: Callable[[], torch.nn.Module] | None = None,
    sliding_model_factory: Callable[[], torch.nn.Module] | None = None,
    embedding_checkpoint_path: str | None = None,
    relevancy_checkpoint_path: str | None = None,
    adapters_checkpoint_path: str | None = None,
) -> list[BenchmarkResult]:
    """
    Benchmark full / sliding / hfold on the SAME dataloader.

    Each mode uses a freshly built model (or caller-supplied factory). Metrics are
    computed independently per mode; no placeholder duplication.
    """
    device_obj = torch.device(device)
    results: list[BenchmarkResult] = []

    if backbone not in {"pythia", "gpt2"}:
        raise ValueError("backbone must be one of: pythia, gpt2")

    def _default_full_factory() -> torch.nn.Module:
        from transformers import AutoModelForCausalLM

        if checkpoint_path and backbone == "pythia":
            return load_gpt_neox_causal_lm_from_folder(checkpoint_path, cache_dir="./data")
        if checkpoint_path:
            return AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                attn_implementation="eager",
            )
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager",
        )

    def _default_sliding_factory() -> torch.nn.Module:
        model = (full_model_factory or _default_full_factory)()
        return _apply_sliding_window(model, sliding_window_size)

    def _build_hfold() -> torch.nn.Module:
        return _build_hfold_model(
            backbone=backbone,
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            config=config,
            embedding_checkpoint_path=embedding_checkpoint_path,
            relevancy_checkpoint_path=relevancy_checkpoint_path,
            adapters_checkpoint_path=adapters_checkpoint_path,
        )

    full_model = (full_model_factory or _default_full_factory)().to(device_obj)
    full_loss, full_tok_s = _run_eval(full_model, dataloader, device_obj)
    results.append(_to_result("full_attention", full_loss, full_tok_s))

    sliding_model = (sliding_model_factory or _default_sliding_factory)().to(device_obj)
    sliding_loss, sliding_tok_s = _run_eval(sliding_model, dataloader, device_obj)
    results.append(_to_result("sliding_window", sliding_loss, sliding_tok_s))

    hfold_model = _build_hfold()
    _move_hfold_bundle_to_device(hfold_model, device_obj)
    hfold_loss, hfold_tok_s = _run_eval(
        hfold_model,
        dataloader,
        device_obj,
        hfold_window_size=sliding_window_size,
        hfold_eval_use_kv_cache=hfold_eval_use_kv_cache,
        hfold_eval_chunk_size=hfold_eval_chunk_size,
    )
    results.append(_to_result("hfold", hfold_loss, hfold_tok_s))

    return results
