from __future__ import annotations

from dataclasses import dataclass

import torch

from ..config.schema import HFoldConfig
from ..models.adapters import BackboneAdapterRegistry
from ..models.interfaces import EmbeddingModelProtocol, RelevancyModelProtocol
from .heap_state import HFoldHeapEntry, HFoldLayerState, HFoldRuntimeState
from .priority_heap import BoundedMaxHeap


@dataclass
class HFoldStepArtifacts:
    popped_entries: list[HFoldHeapEntry]
    evicted_entries: list[HFoldHeapEntry]
    summary_embedding: torch.Tensor | None


class HFoldRuntime:
    """
    Layer-wise runtime for HFold heap folding.
    """

    def __init__(self, config: HFoldConfig) -> None:
        config.validate()
        self.config = config
        self.state = HFoldRuntimeState()
        self._heaps: dict[int, BoundedMaxHeap] = {}
        self._adapters: BackboneAdapterRegistry | None = None
        self._backbone: str | None = None

    def reset(self) -> None:
        self.state = HFoldRuntimeState()
        self._heaps = {}

    def attach_adapters(self, adapters: BackboneAdapterRegistry, backbone: str) -> None:
        if backbone not in adapters.adapters:
            raise ValueError(f"Adapter for backbone '{backbone}' not registered.")
        self._adapters = adapters
        self._backbone = backbone

    def _encode_for_aux_models(self, vectors: torch.Tensor) -> torch.Tensor:
        if self._adapters is None or self._backbone is None:
            return vectors
        return self._adapters.encode(self._backbone, vectors)

    def _decode_from_aux_models(self, vectors: torch.Tensor) -> torch.Tensor:
        if self._adapters is None or self._backbone is None:
            return vectors
        return self._adapters.decode(self._backbone, vectors)

    def _get_layer_state(self, layer_index: int) -> HFoldLayerState:
        if layer_index not in self.state.layers:
            self.state.layers[layer_index] = HFoldLayerState(layer_index=layer_index)
            self._heaps[layer_index] = BoundedMaxHeap(self.config.model.max_heap_size)
        return self.state.layers[layer_index]

    def _build_entry(
        self,
        *,
        layer_state: HFoldLayerState,
        vector: torch.Tensor,
        score: float,
        token_position: int,
        layer_index: int,
        head_index: int,
        time_index: int,
        source: str,
    ) -> HFoldHeapEntry:
        entry = HFoldHeapEntry(
            score=float(score),
            vector=vector.detach().clone(),
            token_position=int(token_position),
            layer_index=layer_index,
            head_index=int(head_index),
            time_index=int(time_index),
            source=source,
            id=layer_state.next_entry_id,
        )
        layer_state.next_entry_id += 1
        return entry

    def prime_timestep_zero(
        self,
        *,
        layer_index: int,
        vectors: torch.Tensor,
        scores: torch.Tensor,
        token_positions: torch.Tensor,
        head_indices: torch.Tensor,
        time_index: int,
    ) -> HFoldStepArtifacts:
        """
        Timestep 0: only insert top-w candidates into heap.
        """
        layer_state = self._get_layer_state(layer_index)
        heap = self._heaps[layer_index]
        entries: list[HFoldHeapEntry] = []
        limit = min(self.config.model.top_w, int(vectors.size(0)))
        for idx in range(limit):
            entries.append(
                self._build_entry(
                    layer_state=layer_state,
                    vector=vectors[idx],
                    score=float(scores[idx].item()),
                    token_position=int(token_positions[idx].item()),
                    layer_index=layer_index,
                    head_index=int(head_indices[idx].item()),
                    time_index=time_index,
                    source="local",
                )
            )
        evicted = heap.push_many(entries)
        layer_state.heap = heap.peek_all()
        self.state.timestep = max(self.state.timestep, time_index)
        return HFoldStepArtifacts(popped_entries=[], evicted_entries=evicted, summary_embedding=None)

    def pop_top_k(self, *, layer_index: int) -> list[HFoldHeapEntry]:
        layer_state = self._get_layer_state(layer_index)
        heap = self._heaps[layer_index]
        popped = heap.pop_top_k(self.config.model.pop_k)
        layer_state.heap = heap.peek_all()
        return popped

    def step_with_reinsert_and_fold(
        self,
        *,
        layer_index: int,
        popped_entries: list[HFoldHeapEntry],
        transformed_popped_vectors: torch.Tensor,
        new_vectors: torch.Tensor,
        new_scores: torch.Tensor,
        new_token_positions: torch.Tensor,
        new_head_indices: torch.Tensor,
        time_index: int,
        embedding_model: EmbeddingModelProtocol,
        relevancy_model: RelevancyModelProtocol,
    ) -> HFoldStepArtifacts:
        layer_state = self._get_layer_state(layer_index)
        heap = self._heaps[layer_index]

        reinsert_entries: list[HFoldHeapEntry] = []
        for idx, popped in enumerate(popped_entries):
            if idx >= transformed_popped_vectors.size(0):
                break
            reinsert_entries.append(
                self._build_entry(
                    layer_state=layer_state,
                    vector=transformed_popped_vectors[idx],
                    score=popped.score,
                    token_position=popped.token_position,
                    layer_index=layer_index,
                    head_index=popped.head_index,
                    time_index=time_index,
                    source="popped_transform",
                )
            )

        local_limit = min(self.config.model.top_w, int(new_vectors.size(0)))
        for idx in range(local_limit):
            reinsert_entries.append(
                self._build_entry(
                    layer_state=layer_state,
                    vector=new_vectors[idx],
                    score=float(new_scores[idx].item()),
                    token_position=int(new_token_positions[idx].item()),
                    layer_index=layer_index,
                    head_index=int(new_head_indices[idx].item()),
                    time_index=time_index,
                    source="local",
                )
            )

        evicted = heap.push_many(reinsert_entries)

        summary = None
        if evicted:
            evicted_tensor = torch.stack([entry.vector for entry in evicted], dim=0).unsqueeze(0)
            padding_mask = torch.ones(1, evicted_tensor.size(1), dtype=torch.bool, device=evicted_tensor.device)
            evicted_latent = self._encode_for_aux_models(evicted_tensor)
            summary = embedding_model.encode_summary(evicted_latent, padding_mask=padding_mask)
            self._fold_current_heap(layer_index=layer_index, summary=summary, relevancy_model=relevancy_model)

        layer_state.heap = heap.peek_all()
        self.state.timestep = max(self.state.timestep, time_index)
        return HFoldStepArtifacts(
            popped_entries=popped_entries,
            evicted_entries=evicted,
            summary_embedding=summary,
        )

    def _fold_current_heap(
        self,
        *,
        layer_index: int,
        summary: torch.Tensor,
        relevancy_model: RelevancyModelProtocol,
    ) -> None:
        layer_state = self._get_layer_state(layer_index)
        heap_entries = self._heaps[layer_index].peek_all()
        if not heap_entries:
            return
        heap_vectors_raw = torch.stack([entry.vector for entry in heap_entries], dim=0).unsqueeze(0)
        heap_vectors_latent = self._encode_for_aux_models(heap_vectors_raw)
        relevancy_scores = relevancy_model.score_heap(summary, heap_vectors_latent)
        scaled_summary = summary.unsqueeze(1)
        updated_latent = heap_vectors_latent + relevancy_scores.unsqueeze(-1) * scaled_summary
        updated_raw = self._decode_from_aux_models(updated_latent)
        for idx, entry in enumerate(heap_entries):
            entry.vector = updated_raw[0, idx].detach().clone()
        # Rebuild heap with updated entries while preserving scores and ids.
        self._heaps[layer_index] = BoundedMaxHeap(self.config.model.max_heap_size)
        self._heaps[layer_index].push_many(heap_entries)
        layer_state.heap = self._heaps[layer_index].peek_all()
