"""Model-level (global-heap) HFold integration.

Per the spec the heap is global across all timesteps and operates AFTER all
layers per timestep:

    timestep 0: forward through all layers -> push top-w from final layer.
    timestep t: pop K, prepend to inputs, forward through all layers,
                push K-transformed + top-w (de-duplicated) from final layer,
                evict, embed-summarize, fold the heap.

This hook replaces the trunk's `forward` so we only do ONE pop and ONE push
per timestep regardless of layer count L.
"""
from __future__ import annotations

from types import MethodType
from typing import Any

import torch

from ..models.interfaces import EmbeddingModelProtocol, RelevancyModelProtocol
from .hfold_runtime import HFoldRuntime
from .vector_store import append_heap_vectors, split_appended_outputs


GLOBAL_HEAP_INDEX = 0


def _expand_attention_mask_for_prepend(
    attention_mask: torch.Tensor | None,
    *,
    heap_len: int,
    new_total: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if attention_mask is None or heap_len == 0:
        return attention_mask
    if attention_mask.dim() == 2:
        batch = attention_mask.size(0)
        prefix = torch.ones(batch, heap_len, dtype=attention_mask.dtype, device=attention_mask.device)
        return torch.cat([prefix, attention_mask.to(device)], dim=1)
    if attention_mask.dim() == 4:
        batch = attention_mask.size(0)
        old_tgt = attention_mask.size(2)
        old_src = attention_mask.size(3)
        target_tgt = max(new_total, old_tgt + heap_len)
        target_src = max(new_total, old_src + heap_len)
        expanded = torch.zeros(batch, 1, target_tgt, target_src, dtype=attention_mask.dtype, device=attention_mask.device)
        expanded[:, :, target_tgt - old_tgt : target_tgt, target_src - old_src : target_src] = attention_mask
        return expanded
    return attention_mask


def _past_kv_length(past_kv: Any) -> int:
    """Return the seq-length already cached in `past_kv`, supporting both the
    legacy tuple-of-tuples format and HF DynamicCache-like objects.
    """
    if past_kv is None:
        return 0
    if isinstance(past_kv, (tuple, list)) and past_kv:
        layer0 = past_kv[0]
        if isinstance(layer0, (tuple, list)) and layer0 and hasattr(layer0[0], "size"):
            return int(layer0[0].size(-2))
    if hasattr(past_kv, "get_seq_length"):
        try:
            return int(past_kv.get_seq_length())
        except Exception:
            return 0
    return 0


def _splice_heap_from_past_kv(past_kv: Any, *, n_before: int, heap_len: int) -> Any:
    """Remove the `heap_len` slots that were inserted at index `n_before` from
    every layer's K/V cache. Heap tokens are appended to the inputs at this
    timestep but must NOT be retained in the cache for future steps.
    """
    if past_kv is None or heap_len <= 0:
        return past_kv
    if isinstance(past_kv, (tuple, list)):
        spliced: list[Any] = []
        for layer in past_kv:
            if (
                isinstance(layer, (tuple, list))
                and len(layer) >= 2
                and hasattr(layer[0], "size")
                and hasattr(layer[1], "size")
            ):
                k_t, v_t = layer[0], layer[1]
                new_k = torch.cat([k_t[..., :n_before, :], k_t[..., n_before + heap_len :, :]], dim=-2)
                new_v = torch.cat([v_t[..., :n_before, :], v_t[..., n_before + heap_len :, :]], dim=-2)
                spliced.append((new_k, new_v))
            else:
                spliced.append(layer)
        return tuple(spliced)
    if hasattr(past_kv, "key_cache") and hasattr(past_kv, "value_cache"):
        for i in range(len(past_kv.key_cache)):
            k_t = past_kv.key_cache[i]
            v_t = past_kv.value_cache[i]
            if k_t is None or v_t is None:
                continue
            past_kv.key_cache[i] = torch.cat(
                [k_t[..., :n_before, :], k_t[..., n_before + heap_len :, :]], dim=-2
            )
            past_kv.value_cache[i] = torch.cat(
                [v_t[..., :n_before, :], v_t[..., n_before + heap_len :, :]], dim=-2
            )
        if hasattr(past_kv, "_seen_tokens"):
            try:
                past_kv._seen_tokens = max(0, int(past_kv._seen_tokens) - heap_len)
            except Exception:
                pass
    return past_kv


def _select_top_candidates(
    attn_weights: torch.Tensor,
    token_vectors: torch.Tensor,
    top_w: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Use the current-query (last query row) attention distribution averaged
    # over heads. This matches autoregressive timestep semantics: select
    # candidates relevant to the token being predicted now.
    score_per_key = attn_weights.mean(dim=1)[:, -1, :]
    k = min(top_w, score_per_key.size(-1), token_vectors.size(1))
    scores, indices = torch.topk(score_per_key[:, : token_vectors.size(1)], k=k, dim=-1)
    gather_index = indices.unsqueeze(-1).expand(-1, -1, token_vectors.size(-1))
    vectors = torch.gather(token_vectors, dim=1, index=gather_index)
    return scores, vectors, indices


class HFoldModelHook:
    """One pop + one push per timestep against a single global heap."""

    def __init__(
        self,
        *,
        embedding_layer: torch.nn.Module,
        runtime: HFoldRuntime,
        embedding_model: EmbeddingModelProtocol,
        relevancy_model: RelevancyModelProtocol,
    ) -> None:
        self.embedding_layer = embedding_layer
        self.runtime = runtime
        self.embedding_model = embedding_model
        self.relevancy_model = relevancy_model

    # ------------------------------------------------------------------ helpers

    def _resolve_embeds(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            return inputs_embeds
        if input_ids is None:
            raise ValueError("HFoldModelHook requires either input_ids or inputs_embeds.")
        return self.embedding_layer(input_ids)

    def _last_attention(self, outputs: Any) -> torch.Tensor:
        attentions = getattr(outputs, "attentions", None)
        if attentions is None or len(attentions) == 0:
            raise ValueError("HFold global hook needs output_attentions=True from the trunk.")
        return attentions[-1]

    def _extract_last_hidden(self, outputs: Any) -> torch.Tensor:
        last_hidden = getattr(outputs, "last_hidden_state", None)
        if last_hidden is None:
            raise ValueError("HFold global hook needs last_hidden_state on trunk outputs.")
        return last_hidden

    # ------------------------------------------------------------------ forward

    def hooked_forward(
        self,
        original_forward,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        input_ids = kwargs.pop("input_ids", None)
        if args:
            input_ids = args[0] if input_ids is None else input_ids
            args = args[1:]
        inputs_embeds = kwargs.pop("inputs_embeds", None)
        attention_mask = kwargs.pop("attention_mask", None)
        kwargs.setdefault("return_dict", True)
        caller_requested_attn = bool(kwargs.get("output_attentions", False))
        cache_keys = ("past_key_values", "past_key_value", "layer_past", "cache")
        prior_past_kv: Any = None
        for cache_key in cache_keys:
            value = kwargs.get(cache_key)
            if value is not None:
                prior_past_kv = value
                break

        # KV cache and sliding window run as the model normally would. The
        # heap tokens are an extra (k) inputs prepended for THIS timestep only;
        # they must not be retained in the past_key_values cache for future
        # steps (we splice them out post-forward). position_ids is dropped so
        # the backbone re-derives positions from `past_kv length + current
        # input length` after we add the heap rows.
        def _build_unified_kwargs() -> dict[str, Any]:
            unified = dict(kwargs)
            unified["output_attentions"] = True
            unified["return_dict"] = True
            unified.pop("position_ids", None)
            return unified

        layer_state = self.runtime._get_layer_state(GLOBAL_HEAP_INDEX)
        timestep = int(layer_state.call_count)

        embeds = self._resolve_embeds(input_ids, inputs_embeds)

        if timestep == 0:
            unified_kwargs = _build_unified_kwargs()
            outputs = original_forward(
                *args,
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                **unified_kwargs,
            )
            outputs_for_heap = outputs
            outputs_to_return = outputs
            last_hidden = self._extract_last_hidden(outputs_for_heap)
            last_attn = self._last_attention(outputs_for_heap)
            scores, vectors, indices = _select_top_candidates(last_attn, last_hidden, self.runtime.config.model.top_w)
            head_indices = torch.zeros(scores.size(-1), dtype=torch.long, device=last_hidden.device)
            self.runtime.prime_timestep_zero(
                layer_index=GLOBAL_HEAP_INDEX,
                vectors=vectors[0],
                scores=scores[0],
                token_positions=indices[0],
                head_indices=head_indices,
                time_index=0,
            )
            layer_state.call_count += 1
            self.runtime.state.timestep = max(self.runtime.state.timestep, layer_state.call_count)
            if not caller_requested_attn and hasattr(outputs_to_return, "attentions"):
                try:
                    outputs_to_return.attentions = None
                except (AttributeError, TypeError):
                    pass
            return outputs_to_return

        seq_len = embeds.size(1)
        popped = self.runtime.pop_top_k(layer_index=GLOBAL_HEAP_INDEX)
        if popped:
            heap_vectors = torch.stack([entry.vector for entry in popped], dim=0).unsqueeze(0).to(embeds.device, embeds.dtype)
        else:
            heap_vectors = embeds.new_zeros((1, 0, embeds.size(-1)))
        heap_len = heap_vectors.size(1)

        augmented_embeds = append_heap_vectors(embeds, heap_vectors)
        new_total = augmented_embeds.size(1)
        augmented_attention_mask = attention_mask
        if augmented_attention_mask is not None:
            augmented_attention_mask = _expand_attention_mask_for_prepend(
                augmented_attention_mask,
                heap_len=heap_len,
                new_total=new_total,
                device=embeds.device,
                dtype=embeds.dtype,
            )

        unified_kwargs = _build_unified_kwargs()
        outputs = original_forward(
            *args,
            inputs_embeds=augmented_embeds,
            attention_mask=augmented_attention_mask,
            **unified_kwargs,
        )
        outputs_for_heap = outputs
        outputs_to_return = outputs

        # Remove heap-token slots from any returned KV cache so future steps
        # only see real-sequence cache entries. Heap was prepended at offset
        # `n_before` (existing cache length) of the new K/V dims.
        if heap_len > 0:
            new_past_kv = getattr(outputs, "past_key_values", None)
            if new_past_kv is None:
                new_past_kv = getattr(outputs, "past_key_value", None)
            if new_past_kv is not None:
                n_before = _past_kv_length(prior_past_kv)
                spliced = _splice_heap_from_past_kv(
                    new_past_kv, n_before=n_before, heap_len=heap_len
                )
                try:
                    outputs_to_return.past_key_values = spliced
                except (AttributeError, TypeError):
                    pass

        last_hidden = self._extract_last_hidden(outputs_for_heap)
        last_attn = self._last_attention(outputs_for_heap)
        token_output, transformed_heap = split_appended_outputs(last_hidden, seq_len)
        if heap_len > 0 and last_attn.dim() == 4:
            attn_for_originals = last_attn[:, :, heap_len:, heap_len:]
        else:
            attn_for_originals = last_attn

        scores, vectors, indices = _select_top_candidates(attn_for_originals, token_output, self.runtime.config.model.top_w)
        head_indices = torch.zeros(scores.size(-1), dtype=torch.long, device=token_output.device)
        artifacts = self.runtime.step_with_reinsert_and_fold(
            layer_index=GLOBAL_HEAP_INDEX,
            popped_entries=popped,
            transformed_popped_vectors=transformed_heap[0] if heap_len > 0 else token_output.new_zeros((0, token_output.size(-1))),
            new_vectors=vectors[0],
            new_scores=scores[0],
            new_token_positions=indices[0],
            new_head_indices=head_indices,
            time_index=timestep,
            embedding_model=self.embedding_model,
            relevancy_model=self.relevancy_model,
        )
        layer_state.call_count += 1
        self.runtime.state.timestep = max(self.runtime.state.timestep, layer_state.call_count)

        # Replace last_hidden_state with token-only slice for the LM head downstream.
        try:
            outputs_to_return.last_hidden_state = token_output
        except (AttributeError, TypeError):
            # Some HF output types are NamedTuple-like; rebuild via dict if possible.
            from dataclasses import is_dataclass, replace

            if is_dataclass(outputs_to_return):
                outputs_to_return = replace(outputs_to_return, last_hidden_state=token_output)
        if not caller_requested_attn and hasattr(outputs_to_return, "attentions"):
            try:
                outputs_to_return.attentions = None
            except (AttributeError, TypeError):
                pass
        return outputs_to_return


def _bind_trunk_forward(trunk: torch.nn.Module, hook: HFoldModelHook) -> None:
    original_forward = trunk.forward

    def patched_forward(_self: torch.nn.Module, *a: Any, **kw: Any) -> Any:
        return hook.hooked_forward(original_forward, *a, **kw)

    trunk.forward = MethodType(patched_forward, trunk)


def wrap_pythia_with_hfold(
    model: torch.nn.Module,
    runtime: HFoldRuntime,
    embedding_model: EmbeddingModelProtocol,
    relevancy_model: RelevancyModelProtocol,
) -> torch.nn.Module:
    trunk = getattr(model, "gpt_neox", None)
    if trunk is None or not hasattr(trunk, "embed_in"):
        raise ValueError("HFold global wrapper requires a Pythia/GPT-NeoX model with `gpt_neox.embed_in`.")
    hook = HFoldModelHook(
        embedding_layer=trunk.embed_in,
        runtime=runtime,
        embedding_model=embedding_model,
        relevancy_model=relevancy_model,
    )
    _bind_trunk_forward(trunk, hook)
    # Expose runtime so callers can reset heap state per sequence/batch.
    model.hfold_runtime = runtime
    # Register as submodules so `model.to(device)` moves them with the backbone.
    model.add_module("hfold_embedding_model", embedding_model)
    model.add_module("hfold_relevancy_model", relevancy_model)
    return model


def wrap_gpt2_with_hfold(
    model: torch.nn.Module,
    runtime: HFoldRuntime,
    embedding_model: EmbeddingModelProtocol,
    relevancy_model: RelevancyModelProtocol,
) -> torch.nn.Module:
    trunk = getattr(model, "transformer", None)
    if trunk is None or not hasattr(trunk, "wte"):
        raise ValueError("HFold global wrapper requires a GPT-2 model with `transformer.wte`.")
    hook = HFoldModelHook(
        embedding_layer=trunk.wte,
        runtime=runtime,
        embedding_model=embedding_model,
        relevancy_model=relevancy_model,
    )
    _bind_trunk_forward(trunk, hook)
    model.hfold_runtime = runtime
    model.add_module("hfold_embedding_model", embedding_model)
    model.add_module("hfold_relevancy_model", relevancy_model)
    return model
