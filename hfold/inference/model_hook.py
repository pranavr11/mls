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


def _select_top_candidates(
    attn_weights: torch.Tensor,
    token_vectors: torch.Tensor,
    top_w: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    score_per_key = attn_weights.mean(dim=1).mean(dim=1)
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
        has_cache_context = bool(kwargs.get("use_cache", False)) or any(
            (key in kwargs and kwargs.get(key) is not None) for key in cache_keys
        )

        def _build_aux_kwargs() -> dict[str, Any]:
            aux_kwargs = dict(kwargs)
            aux_kwargs["output_attentions"] = True
            aux_kwargs["return_dict"] = True
            aux_kwargs["use_cache"] = False
            aux_kwargs.pop("position_ids", None)
            for cache_key in cache_keys:
                aux_kwargs.pop(cache_key, None)
            return aux_kwargs

        def _build_primary_kwargs() -> dict[str, Any]:
            primary_kwargs = dict(kwargs)
            primary_kwargs["output_attentions"] = caller_requested_attn
            primary_kwargs["return_dict"] = True
            return primary_kwargs

        layer_state = self.runtime._get_layer_state(GLOBAL_HEAP_INDEX)
        timestep = int(layer_state.call_count)

        embeds = self._resolve_embeds(input_ids, inputs_embeds)

        if timestep == 0:
            aux_kwargs = _build_aux_kwargs()
            if has_cache_context:
                primary_outputs = original_forward(
                    *args,
                    inputs_embeds=embeds,
                    attention_mask=attention_mask,
                    **_build_primary_kwargs(),
                )
                aux_outputs = original_forward(
                    *args,
                    inputs_embeds=embeds,
                    attention_mask=attention_mask,
                    **aux_kwargs,
                )
                outputs_for_heap = aux_outputs
                outputs_to_return = primary_outputs
            else:
                outputs = original_forward(
                    *args,
                    inputs_embeds=embeds,
                    attention_mask=attention_mask,
                    **aux_kwargs,
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
        primary_attention_mask = attention_mask
        aux_attention_mask = attention_mask
        if aux_attention_mask is not None:
            aux_attention_mask = _expand_attention_mask_for_prepend(
                aux_attention_mask,
                heap_len=heap_len,
                new_total=new_total,
                device=embeds.device,
                dtype=embeds.dtype,
            )

        aux_kwargs = _build_aux_kwargs()
        if has_cache_context:
            primary_outputs = original_forward(
                *args,
                inputs_embeds=embeds,
                attention_mask=primary_attention_mask,
                **_build_primary_kwargs(),
            )
            aux_outputs = original_forward(
                *args,
                inputs_embeds=augmented_embeds,
                attention_mask=aux_attention_mask,
                **aux_kwargs,
            )
            outputs_for_heap = aux_outputs
            outputs_to_return = primary_outputs
        else:
            outputs = original_forward(
                *args,
                inputs_embeds=augmented_embeds,
                attention_mask=aux_attention_mask,
                **aux_kwargs,
            )
            outputs_for_heap = outputs
            outputs_to_return = outputs
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
    return model
