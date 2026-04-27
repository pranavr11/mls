from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..config.schema import HFoldConfig
from ..models.adapters import BackboneAdapterRegistry
from ..models.interfaces import EmbeddingModelProtocol, RelevancyModelProtocol
from .heap_state import HFoldHeapEntry, HFoldLayerState, HFoldRuntimeState
from .tensor_heap import HFoldTensorBundle, pop_top_k_tensor, push_many_tensor


@dataclass
class HFoldStepArtifacts:
    popped_bundle: HFoldTensorBundle | None = None
    evicted_bundle: HFoldTensorBundle | None = None
    summary_embedding: torch.Tensor | None = None
    popped_entries: list[HFoldHeapEntry] = field(default_factory=list)
    evicted_entries: list[HFoldHeapEntry] = field(default_factory=list)


class HFoldRuntime:
    """
    Layer-wise runtime for HFold heap folding.
    """

    def __init__(self, config: HFoldConfig, *, materialize_heap_entries: bool = False) -> None:
        config.validate()
        self.config = config
        self.state = HFoldRuntimeState()
        self._tensor_heaps: dict[int, HFoldTensorBundle] = {}
        self._adapters: BackboneAdapterRegistry | None = None
        self._backbone: str | None = None
        self._materialize_heap_entries = bool(materialize_heap_entries)

    def reset(self) -> None:
        self.state = HFoldRuntimeState()
        self._tensor_heaps = {}

    def attach_adapters(self, adapters: BackboneAdapterRegistry, backbone: str) -> None:
        if backbone not in adapters.adapters:
            raise ValueError(f"Adapter for backbone '{backbone}' not registered.")
        self._adapters = adapters
        self._backbone = backbone

    def set_materialize_heap_entries(self, enabled: bool) -> None:
        self._materialize_heap_entries = bool(enabled)
        if not enabled:
            for layer in self.state.layers.values():
                layer.heap = []
            return
        for layer_index in list(self.state.layers.keys()):
            self._maybe_update_debug_heap(layer_index)

    def _ensure_aux_on_device(
        self,
        device: torch.device,
        embedding_model: EmbeddingModelProtocol,
        relevancy_model: RelevancyModelProtocol,
    ) -> None:
        """Keep adapter + aux MLPs on the same device as backbone activations."""
        if self._adapters is not None:
            self._adapters.to(device)
        if isinstance(embedding_model, torch.nn.Module):
            embedding_model.to(device)
        if isinstance(relevancy_model, torch.nn.Module):
            relevancy_model.to(device)

    def _encode_for_aux_models(self, vectors: torch.Tensor) -> torch.Tensor:
        if self._adapters is None or self._backbone is None:
            return vectors
        self._adapters.to(vectors.device)
        return self._adapters.encode(self._backbone, vectors)

    def _decode_from_aux_models(self, vectors: torch.Tensor) -> torch.Tensor:
        if self._adapters is None or self._backbone is None:
            return vectors
        self._adapters.to(vectors.device)
        return self._adapters.decode(self._backbone, vectors)

    def _get_layer_state(self, layer_index: int) -> HFoldLayerState:
        if layer_index not in self.state.layers:
            self.state.layers[layer_index] = HFoldLayerState(layer_index=layer_index)
            self._tensor_heaps[layer_index] = HFoldTensorBundle.empty(
                hidden_size=int(self.config.model.hidden_size),
            )
        return self.state.layers[layer_index]

    def _next_entry_ids(self, *, layer_state: HFoldLayerState, count: int, device: torch.device) -> torch.Tensor:
        if count <= 0:
            return torch.empty((0,), device=device, dtype=torch.long)
        start = int(layer_state.next_entry_id)
        layer_state.next_entry_id += int(count)
        return torch.arange(start, start + int(count), device=device, dtype=torch.long)

    @staticmethod
    def _concat_bundles(a: HFoldTensorBundle, b: HFoldTensorBundle) -> HFoldTensorBundle:
        if len(a) == 0:
            return b
        if len(b) == 0:
            return a
        return HFoldTensorBundle(
            scores=torch.cat([a.scores, b.scores], dim=0),
            vectors=torch.cat([a.vectors, b.vectors], dim=0),
            token_positions=torch.cat([a.token_positions, b.token_positions], dim=0),
            head_indices=torch.cat([a.head_indices, b.head_indices], dim=0),
            time_indices=torch.cat([a.time_indices, b.time_indices], dim=0),
            entry_ids=torch.cat([a.entry_ids, b.entry_ids], dim=0),
        )

    def _entries_to_bundle(
        self,
        *,
        layer_index: int,
        entries: list[HFoldHeapEntry],
        vector_device: torch.device,
        vector_dtype: torch.dtype,
        score_dtype: torch.dtype,
    ) -> HFoldTensorBundle:
        hidden_size = int(self.config.model.hidden_size)
        if not entries:
            return HFoldTensorBundle.empty(
                hidden_size=hidden_size,
                device=vector_device,
                vector_dtype=vector_dtype,
                score_dtype=score_dtype,
            )
        vectors = torch.stack(
            [entry.vector.detach().to(device=vector_device, dtype=vector_dtype) for entry in entries],
            dim=0,
        )
        return HFoldTensorBundle(
            scores=torch.tensor(
                [float(entry.score) for entry in entries],
                device=vector_device,
                dtype=score_dtype,
            ),
            vectors=vectors,
            token_positions=torch.tensor(
                [int(entry.token_position) for entry in entries],
                device=vector_device,
                dtype=torch.long,
            ),
            head_indices=torch.tensor(
                [int(entry.head_index) for entry in entries],
                device=vector_device,
                dtype=torch.long,
            ),
            time_indices=torch.tensor(
                [int(entry.time_index) for entry in entries],
                device=vector_device,
                dtype=torch.long,
            ),
            entry_ids=torch.tensor(
                [int(entry.id) for entry in entries],
                device=vector_device,
                dtype=torch.long,
            ),
        )

    def _bundle_to_entries(
        self,
        *,
        layer_index: int,
        bundle: HFoldTensorBundle,
        source: str = "tensor",
    ) -> list[HFoldHeapEntry]:
        if len(bundle) == 0:
            return []
        out: list[HFoldHeapEntry] = []
        for idx in range(len(bundle)):
            out.append(
                HFoldHeapEntry(
                    score=float(bundle.scores[idx].item()),
                    vector=bundle.vectors[idx].detach().clone(),
                    token_position=int(bundle.token_positions[idx].item()),
                    layer_index=int(layer_index),
                    head_index=int(bundle.head_indices[idx].item()),
                    time_index=int(bundle.time_indices[idx].item()),
                    source=source,
                    id=int(bundle.entry_ids[idx].item()),
                )
            )
        return out

    def export_heap_entries(self, *, layer_index: int) -> list[HFoldHeapEntry]:
        self._get_layer_state(layer_index)
        return self._bundle_to_entries(
            layer_index=layer_index,
            bundle=self._tensor_heaps[layer_index],
        )

    def _maybe_update_debug_heap(self, layer_index: int) -> None:
        layer_state = self._get_layer_state(layer_index)
        if not self._materialize_heap_entries:
            layer_state.heap = []
            return
        layer_state.heap = self.export_heap_entries(layer_index=layer_index)

    def heap_bundle(self, *, layer_index: int) -> HFoldTensorBundle:
        self._get_layer_state(layer_index)
        return self._tensor_heaps[layer_index]

    def _should_run_aux_fold(self, *, time_index: int) -> bool:
        interval = max(1, int(getattr(self.config.model, "aux_fold_interval", 1)))
        return int(time_index) % interval == 0

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
        limit = min(self.config.model.top_w, int(vectors.size(0)))
        if limit <= 0:
            self._maybe_update_debug_heap(layer_index)
            return HFoldStepArtifacts(
                popped_bundle=HFoldTensorBundle.empty(
                    hidden_size=int(self.config.model.hidden_size),
                    device=vectors.device,
                    vector_dtype=vectors.dtype,
                    score_dtype=scores.dtype,
                ),
                evicted_bundle=HFoldTensorBundle.empty(
                    hidden_size=int(self.config.model.hidden_size),
                    device=vectors.device,
                    vector_dtype=vectors.dtype,
                    score_dtype=scores.dtype,
                ),
                summary_embedding=None,
            )
        candidate_vectors = vectors[:limit].detach().clone()
        candidate_scores = scores[:limit].to(device=candidate_vectors.device, dtype=scores.dtype)
        candidate_positions = token_positions[:limit].to(device=candidate_vectors.device, dtype=torch.long)
        candidate_heads = head_indices[:limit].to(device=candidate_vectors.device, dtype=torch.long)
        candidate_ids = self._next_entry_ids(
            layer_state=layer_state,
            count=limit,
            device=candidate_vectors.device,
        )
        time_indices = torch.full((limit,), int(time_index), device=candidate_vectors.device, dtype=torch.long)
        candidates = HFoldTensorBundle(
            scores=candidate_scores,
            vectors=candidate_vectors,
            token_positions=candidate_positions,
            head_indices=candidate_heads,
            time_indices=time_indices,
            entry_ids=candidate_ids,
        )
        heap, evicted = push_many_tensor(
            heap=self._tensor_heaps[layer_index],
            candidates=candidates,
            capacity=int(self.config.model.max_heap_size),
        )
        self._tensor_heaps[layer_index] = heap
        self._maybe_update_debug_heap(layer_index)
        self.state.timestep = max(self.state.timestep, time_index)
        return HFoldStepArtifacts(
            popped_bundle=HFoldTensorBundle.empty(
                hidden_size=candidates.hidden_size,
                device=candidates.vectors.device,
                vector_dtype=candidates.vectors.dtype,
                score_dtype=candidates.scores.dtype,
            ),
            evicted_bundle=evicted,
            summary_embedding=None,
        )

    def pop_top_k_tensor(self, *, layer_index: int) -> HFoldTensorBundle:
        self._get_layer_state(layer_index)
        heap, popped = pop_top_k_tensor(
            heap=self._tensor_heaps[layer_index],
            k=int(self.config.model.pop_k),
        )
        self._tensor_heaps[layer_index] = heap
        self._maybe_update_debug_heap(layer_index)
        return popped

    def pop_top_k(self, *, layer_index: int) -> list[HFoldHeapEntry]:
        popped = self.pop_top_k_tensor(layer_index=layer_index)
        return self._bundle_to_entries(
            layer_index=layer_index,
            bundle=popped,
            source="pop",
        )

    def step_with_reinsert_and_fold_tensor(
        self,
        *,
        layer_index: int,
        popped_bundle: HFoldTensorBundle,
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
        popped_bundle = popped_bundle.to(
            device=new_vectors.device,
            vector_dtype=new_vectors.dtype,
            score_dtype=new_scores.dtype,
        )
        n_popped = min(len(popped_bundle), int(transformed_popped_vectors.size(0)))
        if n_popped > 0:
            popped_vectors = transformed_popped_vectors[:n_popped].detach().clone()
            popped_scores = popped_bundle.scores[:n_popped]
            popped_positions = popped_bundle.token_positions[:n_popped]
            popped_heads = popped_bundle.head_indices[:n_popped]
            popped_ids = self._next_entry_ids(
                layer_state=layer_state,
                count=n_popped,
                device=popped_vectors.device,
            )
            popped_time = torch.full((n_popped,), int(time_index), device=popped_vectors.device, dtype=torch.long)
            popped_reinsert = HFoldTensorBundle(
                scores=popped_scores,
                vectors=popped_vectors,
                token_positions=popped_positions,
                head_indices=popped_heads,
                time_indices=popped_time,
                entry_ids=popped_ids,
            )
        else:
            popped_positions = torch.empty((0,), device=new_vectors.device, dtype=torch.long)
            popped_reinsert = HFoldTensorBundle.empty(
                hidden_size=int(new_vectors.size(-1)),
                device=new_vectors.device,
                vector_dtype=new_vectors.dtype,
                score_dtype=new_scores.dtype,
            )

        local_limit = min(self.config.model.top_w, int(new_vectors.size(0)))
        if local_limit > 0:
            local_vectors_full = new_vectors[:local_limit]
            local_scores_full = new_scores[:local_limit].to(device=new_vectors.device, dtype=new_scores.dtype)
            local_positions_full = new_token_positions[:local_limit].to(device=new_vectors.device, dtype=torch.long)
            local_heads_full = new_head_indices[:local_limit].to(device=new_vectors.device, dtype=torch.long)
            if popped_positions.numel() > 0:
                keep_mask = ~torch.isin(local_positions_full, popped_positions)
            else:
                keep_mask = torch.ones_like(local_positions_full, dtype=torch.bool)
            local_vectors = local_vectors_full[keep_mask].detach().clone()
            local_scores = local_scores_full[keep_mask]
            local_positions = local_positions_full[keep_mask]
            local_heads = local_heads_full[keep_mask]
            local_count = int(local_vectors.size(0))
            local_ids = self._next_entry_ids(
                layer_state=layer_state,
                count=local_count,
                device=new_vectors.device,
            )
            local_time = torch.full((local_count,), int(time_index), device=new_vectors.device, dtype=torch.long)
            local_reinsert = HFoldTensorBundle(
                scores=local_scores,
                vectors=local_vectors,
                token_positions=local_positions,
                head_indices=local_heads,
                time_indices=local_time,
                entry_ids=local_ids,
            )
        else:
            local_reinsert = HFoldTensorBundle.empty(
                hidden_size=int(new_vectors.size(-1)),
                device=new_vectors.device,
                vector_dtype=new_vectors.dtype,
                score_dtype=new_scores.dtype,
            )

        reinsert = self._concat_bundles(popped_reinsert, local_reinsert)
        heap, evicted = push_many_tensor(
            heap=self._tensor_heaps[layer_index],
            candidates=reinsert,
            capacity=int(self.config.model.max_heap_size),
        )
        self._tensor_heaps[layer_index] = heap

        summary = None
        if len(evicted) > 0 and int(self.config.model.max_heap_size) > 0:
            raw_evicted = evicted.vectors.unsqueeze(0)
            slot_count = int(raw_evicted.size(1))
            target_slots = int(self.config.model.max_heap_size)
            if slot_count < target_slots:
                pad = raw_evicted.new_zeros((1, target_slots - slot_count, raw_evicted.size(-1)))
                evicted_tensor = torch.cat([raw_evicted, pad], dim=1)
            else:
                evicted_tensor = raw_evicted[:, :target_slots, :]
            padding_mask = torch.zeros(1, target_slots, dtype=torch.bool, device=evicted_tensor.device)
            padding_mask[:, : min(slot_count, target_slots)] = True
            if self._should_run_aux_fold(time_index=time_index):
                self._ensure_aux_on_device(evicted_tensor.device, embedding_model, relevancy_model)
                evicted_latent = self._encode_for_aux_models(evicted_tensor)
                summary = embedding_model.encode_summary(evicted_latent, padding_mask=padding_mask)
                self._fold_current_heap(
                    layer_index=layer_index,
                    summary=summary,
                    embedding_model=embedding_model,
                    relevancy_model=relevancy_model,
                )

        self._maybe_update_debug_heap(layer_index)
        self.state.timestep = max(self.state.timestep, time_index)
        return HFoldStepArtifacts(
            popped_bundle=popped_bundle,
            evicted_bundle=evicted,
            summary_embedding=summary,
        )

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
        popped_bundle = self._entries_to_bundle(
            layer_index=layer_index,
            entries=popped_entries,
            vector_device=transformed_popped_vectors.device,
            vector_dtype=transformed_popped_vectors.dtype,
            score_dtype=new_scores.dtype,
        )
        artifacts = self.step_with_reinsert_and_fold_tensor(
            layer_index=layer_index,
            popped_bundle=popped_bundle,
            transformed_popped_vectors=transformed_popped_vectors,
            new_vectors=new_vectors,
            new_scores=new_scores,
            new_token_positions=new_token_positions,
            new_head_indices=new_head_indices,
            time_index=time_index,
            embedding_model=embedding_model,
            relevancy_model=relevancy_model,
        )
        artifacts.popped_entries = list(popped_entries)
        artifacts.evicted_entries = self._bundle_to_entries(
            layer_index=layer_index,
            bundle=artifacts.evicted_bundle
            if artifacts.evicted_bundle is not None
            else HFoldTensorBundle.empty(
                hidden_size=int(self.config.model.hidden_size),
                device=transformed_popped_vectors.device,
                vector_dtype=transformed_popped_vectors.dtype,
                score_dtype=new_scores.dtype,
            ),
            source="evicted",
        )
        return artifacts

    def _fold_current_heap(
        self,
        *,
        layer_index: int,
        summary: torch.Tensor,
        embedding_model: EmbeddingModelProtocol,
        relevancy_model: RelevancyModelProtocol,
    ) -> None:
        self._get_layer_state(layer_index)
        heap_bundle = self._tensor_heaps[layer_index]
        if len(heap_bundle) == 0:
            return
        heap_vectors_raw = heap_bundle.vectors.unsqueeze(0)
        self._ensure_aux_on_device(heap_vectors_raw.device, embedding_model, relevancy_model)
        heap_vectors_latent = self._encode_for_aux_models(heap_vectors_raw)
        # Relevancy model scores in adapter space. Decode one slot from the
        # bottleneck summary to recover adapter-space summary features.
        if hasattr(embedding_model, "decode_from_summary"):
            summary_for_relevancy = embedding_model.decode_from_summary(summary, num_slots=1).squeeze(1)
        else:
            # Backward-compatible fallback for minimal test doubles.
            summary_for_relevancy = summary
        relevancy_scores = relevancy_model.score_heap(summary_for_relevancy, heap_vectors_latent)
        # Spec-aligned fold in backbone space: h_i <- h_i + r_i * g_raw.
        # We keep relevancy scoring in latent space but decode the summary
        # once and apply the additive update on raw backbone vectors.
        summary_raw = self._decode_from_aux_models(summary_for_relevancy.unsqueeze(1)).squeeze(1)
        updated_raw = heap_vectors_raw + relevancy_scores.unsqueeze(-1) * summary_raw.unsqueeze(1)
        self._tensor_heaps[layer_index] = HFoldTensorBundle(
            scores=heap_bundle.scores,
            vectors=updated_raw[0].detach().clone(),
            token_positions=heap_bundle.token_positions,
            head_indices=heap_bundle.head_indices,
            time_indices=heap_bundle.time_indices,
            entry_ids=heap_bundle.entry_ids,
        )
        self._maybe_update_debug_heap(layer_index)
