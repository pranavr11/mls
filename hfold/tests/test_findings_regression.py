"""Regression tests for the external-review findings (H6, H7, H8, H9, H10)."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from hfold.config.schema import HFoldConfig, HFoldModelConfig
from hfold.inference.hfold_runtime import HFoldRuntime
from hfold.inference.model_hook import (
    GLOBAL_HEAP_INDEX,
    _expand_attention_mask_for_prepend,
    _select_top_candidates,
    wrap_pythia_with_hfold,
)
from hfold.integration.benchmark_runner import benchmark_three_modes
from hfold.models.adapters import BackboneAdapterRegistry
from hfold.models.embedding_autoencoder import EmbeddingAutoencoder
from hfold.models.relevancy_transformer import RelevancyTransformer


def test_select_top_candidates_returns_actual_indices():
    attn = torch.zeros(1, 1, 3, 4)
    attn[0, 0, :, 2] = 1.0  # key index 2 dominates
    attn[0, 0, :, 0] = 0.5
    token_vectors = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    scores, vectors, indices = _select_top_candidates(attn, token_vectors, top_w=2)
    assert indices.shape == (1, 2)
    sorted_indices = sorted(indices[0].tolist())
    assert sorted_indices == [0, 2]


def test_expand_attention_mask_2d_prepends_ones():
    mask = torch.tensor([[1, 1, 1]], dtype=torch.long)
    expanded = _expand_attention_mask_for_prepend(
        mask, heap_len=2, new_total=5, device=torch.device("cpu"), dtype=torch.float32
    )
    assert expanded.shape == (1, 5)
    assert expanded[0, :2].tolist() == [1, 1]


def test_expand_attention_mask_4d_keeps_original_block():
    mask = torch.full((1, 1, 3, 3), -1e9)
    triu = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
    mask[0, 0][triu] = -1e9
    mask[0, 0][~triu] = 0.0
    expanded = _expand_attention_mask_for_prepend(
        mask, heap_len=2, new_total=5, device=torch.device("cpu"), dtype=mask.dtype
    )
    assert expanded.shape == (1, 1, 5, 5)
    assert torch.allclose(expanded[0, 0, 2:, 2:], mask[0, 0])
    assert torch.allclose(expanded[0, 0, :2, :], torch.zeros(2, 5))
    assert torch.allclose(expanded[0, 0, :, :2], torch.zeros(5, 2))


@dataclass
class _MaskAwareTrunkOutput:
    last_hidden_state: torch.Tensor
    attentions: tuple[torch.Tensor, ...]
    last_seen_attention_mask_shape: tuple[int, ...] | None = None
    last_seen_inputs_embeds_shape: tuple[int, ...] | None = None


class _MaskAwarePythiaTrunk(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int = 32) -> None:
        super().__init__()
        self.embed_in = nn.Embedding(vocab_size, hidden_size)
        self.layer = nn.Linear(hidden_size, hidden_size)
        self.last_seen_attention_mask_shape: tuple[int, ...] | None = None
        self.last_seen_inputs_embeds_shape: tuple[int, ...] | None = None
        self.last_seen_position_ids_shape: tuple[int, ...] | None = None

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
        position_ids = _kwargs.get("position_ids")
        h = inputs_embeds if inputs_embeds is not None else self.embed_in(input_ids)
        self.last_seen_inputs_embeds_shape = tuple(int(x) for x in h.shape)
        self.last_seen_attention_mask_shape = (
            None if attention_mask is None else tuple(int(x) for x in attention_mask.shape)
        )
        self.last_seen_position_ids_shape = (
            None if position_ids is None else tuple(int(x) for x in position_ids.shape)
        )
        b, s, _ = h.shape
        h = self.layer(h)
        attns = (torch.softmax(torch.zeros(b, 1, s, s, device=h.device), dim=-1),) if output_attentions else tuple()
        return _MaskAwareTrunkOutput(last_hidden_state=h, attentions=attns)


class _MaskAwarePythia(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.gpt_neox = _MaskAwarePythiaTrunk(hidden_size)


def test_global_hook_aligns_attention_mask_after_prepend():
    torch.manual_seed(0)
    hidden_size = 8
    config = HFoldConfig(
        model=HFoldModelConfig(
            hidden_size=hidden_size,
            num_heads=2,
            max_heap_size=2,
            top_w=2,
            pop_k=2,
            adapter_dim=8,
        )
    )
    runtime = HFoldRuntime(config)
    runtime.attach_adapters(
        BackboneAdapterRegistry(specs={"pythia": hidden_size}, shared_dim=8),
        "pythia",
    )
    model = _MaskAwarePythia(hidden_size)
    embed = EmbeddingAutoencoder(hidden_size=8, latent_size=8, max_slots=config.model.max_heap_size)
    rel = RelevancyTransformer(hidden_size=8, num_layers=1, num_heads=2)
    wrap_pythia_with_hfold(model, runtime, embed, rel)

    seq_len = 3
    input_ids = torch.randint(0, 16, (1, seq_len))
    attention_mask = torch.zeros(1, 1, seq_len, seq_len)
    _ = model.gpt_neox(input_ids=input_ids, attention_mask=attention_mask)
    _ = model.gpt_neox(input_ids=input_ids, attention_mask=attention_mask)

    trunk = model.gpt_neox
    assert trunk.last_seen_inputs_embeds_shape is not None
    assert trunk.last_seen_attention_mask_shape is not None
    assert trunk.last_seen_inputs_embeds_shape[1] == 5  # 3 + K=2
    assert trunk.last_seen_attention_mask_shape[-1] == 5
    assert trunk.last_seen_attention_mask_shape[-2] == 5


def test_global_hook_drops_position_ids_for_augmented_aux_pass():
    torch.manual_seed(0)
    hidden_size = 8
    config = HFoldConfig(
        model=HFoldModelConfig(
            hidden_size=hidden_size,
            num_heads=2,
            max_heap_size=2,
            top_w=2,
            pop_k=2,
            adapter_dim=8,
        )
    )
    runtime = HFoldRuntime(config)
    runtime.attach_adapters(
        BackboneAdapterRegistry(specs={"pythia": hidden_size}, shared_dim=8),
        "pythia",
    )
    model = _MaskAwarePythia(hidden_size)
    embed = EmbeddingAutoencoder(hidden_size=8, latent_size=8, max_slots=config.model.max_heap_size)
    rel = RelevancyTransformer(hidden_size=8, num_layers=1, num_heads=2)
    wrap_pythia_with_hfold(model, runtime, embed, rel)

    input_ids = torch.randint(0, 16, (1, 3))
    _ = model.gpt_neox(
        input_ids=input_ids,
        position_ids=torch.arange(0, 3).unsqueeze(0),
        use_cache=True,
    )
    _ = model.gpt_neox(
        input_ids=input_ids[:, :1],
        position_ids=torch.tensor([[3]]),
        use_cache=True,
    )

    trunk = model.gpt_neox
    assert trunk.last_seen_inputs_embeds_shape is not None
    assert trunk.last_seen_inputs_embeds_shape[1] == 3  # 1 + K=2 on aux pass
    assert trunk.last_seen_position_ids_shape is None


def test_runtime_dedupes_popped_token_positions_from_top_w():
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
    runtime.step_with_reinsert_and_fold(
        layer_index=GLOBAL_HEAP_INDEX,
        popped_entries=popped,
        transformed_popped_vectors=pop_vec.clone(),
        new_vectors=pop_vec.clone(),
        new_scores=torch.tensor([0.95, 0.85]),
        new_token_positions=torch.tensor([popped[0].token_position, 99]),
        new_head_indices=torch.tensor([0, 0]),
        time_index=1,
        embedding_model=embed,
        relevancy_model=rel,
    )
    heap = runtime.state.layers[GLOBAL_HEAP_INDEX].heap
    positions = [entry.token_position for entry in heap]
    assert popped[0].token_position in positions
    assert positions.count(popped[0].token_position) == 1


class _ConstantLossModel(nn.Module):
    def __init__(self, vocab_size: int, scale: float) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 8)
        self.head = nn.Linear(8, vocab_size)
        self.scale = scale

    def forward(self, input_ids, attention_mask=None, labels=None):
        del attention_mask
        hidden = self.embed(input_ids) * self.scale
        logits = self.head(hidden)
        loss = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
        )
        return type("Out", (), {"loss": loss, "logits": logits})()


def test_benchmark_three_modes_returns_distinct_results():
    torch.manual_seed(0)

    def collate(_samples):
        return {
            "input_ids": torch.randint(0, 16, (1, 6)),
            "attention_mask": torch.ones(1, 6, dtype=torch.long),
            "labels": torch.randint(0, 16, (1, 6)),
        }

    dataloader = [collate(None) for _ in range(2)]
    config = HFoldConfig(model=HFoldModelConfig(hidden_size=8, num_heads=2, max_heap_size=2, adapter_dim=8))

    def full_factory():
        return _ConstantLossModel(vocab_size=16, scale=1.0)

    def sliding_factory():
        return _ConstantLossModel(vocab_size=16, scale=2.0)

    def hfold_build():
        return _ConstantLossModel(vocab_size=16, scale=3.0)

    from hfold.integration import benchmark_runner

    original_build_pythia = benchmark_runner.build_pythia_with_hfold
    benchmark_runner.build_pythia_with_hfold = lambda **kwargs: type("Bundle", (), {"model": hfold_build()})  # type: ignore
    try:
        results = benchmark_three_modes(
            backbone="pythia",
            model_name="dummy",
            checkpoint_path=None,
            dataloader=dataloader,
            config=config,
            full_model_factory=full_factory,
            sliding_model_factory=sliding_factory,
        )
    finally:
        benchmark_runner.build_pythia_with_hfold = original_build_pythia

    losses = {r.mode: r.loss for r in results}
    assert set(losses) == {"full_attention", "sliding_window", "hfold"}
    assert len({round(losses[m], 6) for m in losses}) == 3
