"""Runtime probes for external-review findings (debug session 01ddb9).

H6: attention mask shape mismatch after prepend.
H7: autoencoder broadcast collapses distinct slots.
H8: reinsertion does not de-duplicate K-popped vectors from top-w candidates.
H9: benchmark_three_modes returns identical metrics for all three modes.
"""
from __future__ import annotations

import torch
from torch import nn

from dataclasses import dataclass

from hfold._debug_log import debug_log
from hfold.config.schema import HFoldConfig, HFoldModelConfig
from hfold.inference.hfold_runtime import HFoldRuntime
from hfold.inference.model_hook import GLOBAL_HEAP_INDEX, wrap_pythia_with_hfold
from hfold.models.adapters import BackboneAdapterRegistry
from hfold.models.embedding_autoencoder import EmbeddingAutoencoder
from hfold.models.relevancy_transformer import RelevancyTransformer


@dataclass
class _TrunkOutput:
    last_hidden_state: torch.Tensor
    attentions: tuple[torch.Tensor, ...]


class HFAwareTrunk(nn.Module):
    """HF-style Pythia-like trunk that records (hidden, mask) shapes for H6 evidence."""

    def __init__(self, hidden_size: int, vocab_size: int = 32) -> None:
        super().__init__()
        self.embed_in = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        output_attentions=False,
        return_dict=True,
        **_kwargs,
    ):
        del return_dict
        h = inputs_embeds if inputs_embeds is not None else self.embed_in(input_ids)
        b, s, _ = h.shape
        debug_log(
            hypothesis_id="H6",
            location="hfold/scripts/probe_findings.py:HFAwareTrunk.forward",
            message="shapes seen by HF-style trunk",
            data={
                "hidden_shape": [int(x) for x in h.shape],
                "mask_shape": None if attention_mask is None else [int(x) for x in attention_mask.shape],
            },
        )
        h = self.proj(h)
        attns = (torch.softmax(torch.zeros(b, 1, s, s, device=h.device), dim=-1),) if output_attentions else tuple()
        return _TrunkOutput(last_hidden_state=h, attentions=attns)


class TwoLayerHFLike(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.gpt_neox = HFAwareTrunk(hidden_size)


def probe_h6_mask_shape_mismatch():
    torch.manual_seed(0)
    hidden_size = 8
    config = HFoldConfig(
        model=HFoldModelConfig(
            hidden_size=hidden_size,
            num_heads=2,
            max_heap_size=2,
            top_w=2,
            pop_k=2,
            adapter_dim=16,
        )
    )
    runtime = HFoldRuntime(config)
    runtime.attach_adapters(
        BackboneAdapterRegistry(specs={"pythia": hidden_size}, shared_dim=16),
        "pythia",
    )
    model = TwoLayerHFLike(hidden_size)
    embed = EmbeddingAutoencoder(hidden_size=16, latent_size=16, max_slots=config.model.max_heap_size)
    rel = RelevancyTransformer(hidden_size=16, num_layers=1, num_heads=2)
    wrap_pythia_with_hfold(model, runtime, embed, rel)

    seq_len = 3
    input_ids = torch.randint(0, 16, (1, seq_len))
    attention_mask = torch.ones(1, 1, seq_len, seq_len, dtype=torch.float32)
    _ = model.gpt_neox(input_ids=input_ids, attention_mask=attention_mask)
    _ = model.gpt_neox(input_ids=input_ids, attention_mask=attention_mask)


def probe_h7_autoencoder_slot_collapse():
    torch.manual_seed(0)
    model = EmbeddingAutoencoder(hidden_size=8, latent_size=8, max_slots=4)
    distinct = torch.arange(4 * 8, dtype=torch.float32).reshape(1, 4, 8) * 0.05
    reconstructed, _ = model(distinct)
    diffs = []
    for i in range(reconstructed.size(1)):
        for j in range(reconstructed.size(1)):
            if i == j:
                continue
            diffs.append(float((reconstructed[0, i] - reconstructed[0, j]).abs().max().item()))
    debug_log(
        hypothesis_id="H7",
        location="hfold/scripts/probe_findings.py:probe_h7_autoencoder_slot_collapse",
        message="max abs diff between any two reconstructed slots",
        data={
            "max_pairwise_diff": max(diffs),
            "num_pairs": len(diffs),
            "first_slot_first_5": [float(v) for v in reconstructed[0, 0, :5].tolist()],
            "second_slot_first_5": [float(v) for v in reconstructed[0, 1, :5].tolist()],
        },
    )


def probe_h8_reinsertion_duplicates():
    """Force a scenario where the wrapper would feed the same vector through both
    'popped transform' (post-attention K) and 'local top-w' (original sequence)
    paths, and check the heap entries for vector duplication.
    """
    torch.manual_seed(0)
    hidden_size = 4
    config = HFoldConfig(
        model=HFoldModelConfig(
            hidden_size=hidden_size,
            num_heads=2,
            max_heap_size=4,
            top_w=2,
            pop_k=1,
            adapter_dim=8,
        )
    )
    runtime = HFoldRuntime(config)
    runtime.attach_adapters(
        BackboneAdapterRegistry(specs={"pythia": hidden_size}, shared_dim=8),
        "pythia",
    )
    embed = EmbeddingAutoencoder(hidden_size=8, latent_size=8, max_slots=config.model.max_heap_size)
    rel = RelevancyTransformer(hidden_size=8, num_layers=1, num_heads=2)

    runtime.prime_timestep_zero(
        layer_index=GLOBAL_HEAP_INDEX,
        vectors=torch.tensor([[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]]),
        scores=torch.tensor([0.9, 0.5]),
        token_positions=torch.tensor([0, 1]),
        head_indices=torch.tensor([0, 0]),
        time_index=0,
    )
    popped = runtime.pop_top_k(layer_index=GLOBAL_HEAP_INDEX)
    pop_vec = torch.stack([entry.vector for entry in popped], dim=0)
    transformed = pop_vec.clone()
    new_vectors = pop_vec.clone()
    new_scores = torch.tensor([0.95, 0.85])
    new_positions = torch.tensor([0, 1])
    new_heads = torch.tensor([0, 0])
    runtime.step_with_reinsert_and_fold(
        layer_index=GLOBAL_HEAP_INDEX,
        popped_entries=popped,
        transformed_popped_vectors=transformed,
        new_vectors=new_vectors,
        new_scores=new_scores,
        new_token_positions=new_positions,
        new_head_indices=new_heads,
        time_index=1,
        embedding_model=embed,
        relevancy_model=rel,
    )
    heap = runtime.state.layers[GLOBAL_HEAP_INDEX].heap
    duplicate_position_pairs = 0
    for i in range(len(heap)):
        for j in range(i + 1, len(heap)):
            if heap[i].token_position == heap[j].token_position:
                duplicate_position_pairs += 1
    debug_log(
        hypothesis_id="H8",
        location="hfold/scripts/probe_findings.py:probe_h8_reinsertion_duplicates",
        message="heap content after reinsertion with overlapping popped & local",
        data={
            "heap_size": len(heap),
            "heap_token_positions": [int(e.token_position) for e in heap],
            "heap_sources": [str(e.source) for e in heap],
            "duplicate_position_pairs": duplicate_position_pairs,
        },
    )


def probe_h9_benchmark_placeholder():
    from hfold.integration.benchmark_runner import BenchmarkResult, benchmark_three_modes  # noqa: F401
    debug_log(
        hypothesis_id="H9",
        location="hfold/scripts/probe_findings.py:probe_h9_benchmark_placeholder",
        message="static analysis: benchmark_three_modes returns identical loss/tok_s for all 3 modes",
        data={
            "evidence": "benchmark_runner.py builds one model and reuses 'loss, tok_s' for full and sliding modes",
        },
    )


def main() -> None:
    probe_h6_mask_shape_mismatch()
    probe_h7_autoencoder_slot_collapse()
    probe_h8_reinsertion_duplicates()
    probe_h9_benchmark_placeholder()


if __name__ == "__main__":
    main()
