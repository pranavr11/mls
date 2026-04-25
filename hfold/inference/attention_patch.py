from __future__ import annotations

from types import MethodType
from typing import Any

import torch

from ..models.interfaces import EmbeddingModelProtocol, RelevancyModelProtocol
from .hfold_runtime import HFoldRuntime
from .vector_store import append_heap_vectors, split_appended_outputs


def _select_top_candidates(attn_weights: torch.Tensor, token_vectors: torch.Tensor, top_w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    attn_weights: [batch, heads, query, key]
    token_vectors: [batch, seq, hidden]
    returns (scores, vectors) with shape [batch, top_w], [batch, top_w, hidden]
    """
    score_per_key = attn_weights.mean(dim=1).mean(dim=1)
    k = min(top_w, score_per_key.size(-1), token_vectors.size(1))
    scores, indices = torch.topk(score_per_key[:, : token_vectors.size(1)], k=k, dim=-1)
    gather_index = indices.unsqueeze(-1).expand(-1, -1, token_vectors.size(-1))
    vectors = torch.gather(token_vectors, dim=1, index=gather_index)
    return scores, vectors


class HFoldAttentionWrapper(torch.nn.Module):
    def __init__(
        self,
        *,
        original_attention: torch.nn.Module,
        runtime: HFoldRuntime,
        embedding_model: EmbeddingModelProtocol,
        relevancy_model: RelevancyModelProtocol,
        layer_index: int,
    ) -> None:
        super().__init__()
        self.original_attention = original_attention
        self.runtime = runtime
        self.embedding_model = embedding_model
        self.relevancy_model = relevancy_model
        self.layer_index = layer_index

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        if hidden_states.dim() != 3:
            return self.original_attention(hidden_states, *args, **kwargs)

        batch_size, seq_len, hidden_size = hidden_states.shape
        if batch_size != 1:
            # keep conservative behavior for now
            return self.original_attention(hidden_states, *args, **kwargs)

        kwargs = dict(kwargs)
        kwargs["output_attentions"] = True
        layer_state = self.runtime._get_layer_state(self.layer_index)
        timestep = int(layer_state.call_count)

        if timestep == 0:
            outputs = self.original_attention(hidden_states, *args, **kwargs)
            attn_output = outputs[0]
            attn_weights = outputs[-1]
            scores, vectors = _select_top_candidates(attn_weights, attn_output, self.runtime.config.model.top_w)
            top_k = scores.size(-1)
            token_positions = torch.arange(top_k, device=hidden_states.device, dtype=torch.long)
            head_indices = torch.zeros(top_k, device=hidden_states.device, dtype=torch.long)
            self.runtime.prime_timestep_zero(
                layer_index=self.layer_index,
                vectors=vectors[0],
                scores=scores[0],
                token_positions=token_positions,
                head_indices=head_indices,
                time_index=0,
            )
            layer_state.call_count += 1
            self.runtime.state.timestep = max(self.runtime.state.timestep, layer_state.call_count)
            return outputs

        popped = self.runtime.pop_top_k(layer_index=self.layer_index)
        if popped:
            heap_vectors = torch.stack([entry.vector for entry in popped], dim=0).unsqueeze(0).to(hidden_states.device, hidden_states.dtype)
        else:
            heap_vectors = hidden_states.new_zeros((1, 0, hidden_size))

        heap_len = heap_vectors.size(1)
        augmented_hidden = append_heap_vectors(hidden_states, heap_vectors)
        outputs = self.original_attention(augmented_hidden, *args, **kwargs)
        attn_output = outputs[0]
        token_output, transformed_heap = split_appended_outputs(attn_output, seq_len)
        attn_weights = outputs[-1]
        if isinstance(attn_weights, torch.Tensor) and heap_len > 0 and attn_weights.dim() == 4:
            attn_weights_for_originals = attn_weights[:, :, heap_len:, heap_len:]
        else:
            attn_weights_for_originals = attn_weights
        scores, vectors = _select_top_candidates(attn_weights_for_originals, token_output, self.runtime.config.model.top_w)
        top_k = scores.size(-1)
        token_positions = torch.arange(top_k, device=hidden_states.device, dtype=torch.long)
        head_indices = torch.zeros(top_k, device=hidden_states.device, dtype=torch.long)
        self.runtime.step_with_reinsert_and_fold(
            layer_index=self.layer_index,
            popped_entries=popped,
            transformed_popped_vectors=transformed_heap[0],
            new_vectors=vectors[0],
            new_scores=scores[0],
            new_token_positions=token_positions,
            new_head_indices=head_indices,
            time_index=timestep,
            embedding_model=self.embedding_model,
            relevancy_model=self.relevancy_model,
        )
        layer_state.call_count += 1
        self.runtime.state.timestep = max(self.runtime.state.timestep, layer_state.call_count)

        # Replace attention output with token-only projection to preserve shape contract.
        output_as_list = list(outputs)
        output_as_list[0] = token_output
        return tuple(output_as_list)


def _bind_wrapper_forward(module: torch.nn.Module, wrapper: HFoldAttentionWrapper) -> None:
    def patched_forward(_self: torch.nn.Module, hidden_states: torch.Tensor, *a: Any, **kw: Any) -> Any:
        return wrapper(hidden_states, *a, **kw)

    module.forward = MethodType(patched_forward, module)


def patch_pythia_model_attention(
    model: torch.nn.Module,
    runtime: HFoldRuntime,
    embedding_model: EmbeddingModelProtocol,
    relevancy_model: RelevancyModelProtocol,
) -> torch.nn.Module:
    layer_index = 0
    for _, module in model.named_modules():
        if hasattr(module, "query_key_value") and hasattr(module, "dense") and callable(getattr(module, "forward", None)):
            module.attn_hfold_original_forward = module.forward
            wrapper = HFoldAttentionWrapper(
                original_attention=module.attn_hfold_original_forward,
                runtime=runtime,
                embedding_model=embedding_model,
                relevancy_model=relevancy_model,
                layer_index=layer_index,
            )
            _bind_wrapper_forward(module, wrapper)
            layer_index += 1
    return model


def patch_gpt2_model_attention(
    model: torch.nn.Module,
    runtime: HFoldRuntime,
    embedding_model: EmbeddingModelProtocol,
    relevancy_model: RelevancyModelProtocol,
) -> torch.nn.Module:
    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        raise ValueError("GPT-2 model does not expose transformer.h blocks.")
    for layer_index, block in enumerate(model.transformer.h):
        attn = block.attn
        attn.attn_hfold_original_forward = attn.forward
        wrapper = HFoldAttentionWrapper(
            original_attention=attn.attn_hfold_original_forward,
            runtime=runtime,
            embedding_model=embedding_model,
            relevancy_model=relevancy_model,
            layer_index=layer_index,
        )
        _bind_wrapper_forward(attn, wrapper)
    return model
