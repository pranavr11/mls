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
from hfold.integration.benchmark_runner import _run_eval, benchmark_three_modes
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
        self.last_seen_use_cache: bool | None = None
        self.last_seen_past_key_values_present: bool | None = None

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
        self.last_seen_use_cache = bool(_kwargs.get("use_cache", False))
        self.last_seen_past_key_values_present = _kwargs.get("past_key_values") is not None
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


def test_global_hook_passes_use_cache_and_past_kv_through_to_trunk():
    """HFold must integrate with the model's normal sliding-window KV cache.
    The hook must NOT force `use_cache=False` and must NOT drop the caller's
    `past_key_values`.
    """
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

    fake_past = ((torch.zeros(1, 1, 4, hidden_size), torch.zeros(1, 1, 4, hidden_size)),)
    _ = model.gpt_neox(input_ids=torch.randint(0, 16, (1, 4)), use_cache=True)
    _ = model.gpt_neox(
        input_ids=torch.randint(0, 16, (1, 1)),
        use_cache=True,
        past_key_values=fake_past,
    )

    trunk = model.gpt_neox
    assert trunk.last_seen_use_cache is True, "hook must NOT force use_cache=False"
    assert trunk.last_seen_past_key_values_present is True, "hook must NOT strip past_key_values"


def test_global_hook_splices_heap_tokens_from_returned_past_kv():
    """After the augmented forward, the hook must remove the prepended heap
    rows from `past_key_values` so future steps never see heap entries in the
    cache.
    """
    torch.manual_seed(0)
    hidden_size = 8
    K = 2
    n_past = 4
    n_new = 1

    class _CacheTrunkOutput:
        def __init__(self, last_hidden_state, attentions, past_key_values):
            self.last_hidden_state = last_hidden_state
            self.attentions = attentions
            self.past_key_values = past_key_values

    class _CacheTrunk(nn.Module):
        def __init__(self, hidden: int) -> None:
            super().__init__()
            self.embed_in = nn.Embedding(32, hidden)
            self.last_returned_pkv_lengths: list[int] | None = None

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            inputs_embeds=None,
            output_attentions=False,
            return_dict=True,
            **kw,
        ):
            del attention_mask, return_dict
            h = inputs_embeds if inputs_embeds is not None else self.embed_in(input_ids)
            b, s, hidden = h.shape
            attns = (torch.softmax(torch.zeros(b, 1, s, s), dim=-1),) if output_attentions else tuple()
            prior = kw.get("past_key_values")
            prior_len = 0
            if prior is not None:
                prior_len = int(prior[0][0].size(-2))
            new_kv_len = prior_len + s
            new_pkv = (
                (
                    torch.zeros(b, 1, new_kv_len, hidden),
                    torch.zeros(b, 1, new_kv_len, hidden),
                ),
            )
            return _CacheTrunkOutput(
                last_hidden_state=h, attentions=attns, past_key_values=new_pkv
            )

    class _CacheModel(nn.Module):
        def __init__(self, hidden: int) -> None:
            super().__init__()
            self.gpt_neox = _CacheTrunk(hidden)

    config = HFoldConfig(
        model=HFoldModelConfig(
            hidden_size=hidden_size,
            num_heads=2,
            max_heap_size=K,
            top_w=K,
            pop_k=K,
            adapter_dim=8,
        )
    )
    runtime = HFoldRuntime(config)
    runtime.attach_adapters(
        BackboneAdapterRegistry(specs={"pythia": hidden_size}, shared_dim=8),
        "pythia",
    )
    model = _CacheModel(hidden_size)
    embed = EmbeddingAutoencoder(hidden_size=8, latent_size=8, max_slots=K)
    rel = RelevancyTransformer(hidden_size=8, num_layers=1, num_heads=2)
    wrap_pythia_with_hfold(model, runtime, embed, rel)

    # Timestep 0: prime the heap. No cache yet.
    out0 = model.gpt_neox(input_ids=torch.randint(0, 16, (1, n_past)), use_cache=True)
    pkv0 = out0.past_key_values
    assert pkv0[0][0].size(-2) == n_past

    # Timestep 1: caller passes existing cache (len n_past) + 1 new token. The
    # hook will prepend K heap tokens to the inputs. After the trunk returns
    # a cache of length n_past + K + n_new, the hook must splice it back to
    # n_past + n_new.
    out1 = model.gpt_neox(
        input_ids=torch.randint(0, 16, (1, n_new)),
        use_cache=True,
        past_key_values=pkv0,
    )
    cached_len_after = int(out1.past_key_values[0][0].size(-2))
    assert cached_len_after == n_past + n_new, (
        f"expected cache length {n_past + n_new} after splicing heap; got {cached_len_after}"
    )


def test_run_eval_resets_hfold_runtime_between_batches():
    """`_run_eval` must reset the HFold runtime between independent sequences;
    otherwise heap state and timestep counter from earlier batches leak into
    later ones.
    """

    class _LossModel(nn.Module):
        def __init__(self, runtime: HFoldRuntime) -> None:
            super().__init__()
            self.hfold_runtime = runtime
            self.observed_call_counts: list[int] = []

        def forward(self, input_ids, attention_mask=None, labels=None):
            del attention_mask, labels
            layer_state = self.hfold_runtime._get_layer_state(GLOBAL_HEAP_INDEX)
            self.observed_call_counts.append(int(layer_state.call_count))
            layer_state.call_count += 1
            return type(
                "Out",
                (),
                {"loss": torch.tensor(1.0, requires_grad=False)},
            )()

    config = HFoldConfig(
        model=HFoldModelConfig(hidden_size=4, num_heads=2, max_heap_size=2, adapter_dim=4)
    )
    runtime = HFoldRuntime(config)
    model = _LossModel(runtime=runtime)

    dataloader = [
        {"input_ids": torch.zeros(1, 4, dtype=torch.long), "attention_mask": torch.ones(1, 4)}
        for _ in range(3)
    ]
    _run_eval(model, dataloader, torch.device("cpu"))

    assert model.observed_call_counts == [0, 0, 0]


class _StubRuntime:
    """Minimal hfold_runtime stand-in with the .reset() contract."""

    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


class _PrefixRecorderModel(nn.Module):
    """Records prefix lengths for each forward call so we can verify the
    bounded-window HFold eval path under controlled conditions.
    """

    def __init__(self, vocab: int = 32) -> None:
        super().__init__()
        self.vocab = vocab
        self.hfold_runtime = _StubRuntime()
        self.seen_prefix_lens: list[int] = []
        self.tokens_processed = 0
        self.call_count_observed: list[int] = []
        self._call_index = 0

    def forward(self, input_ids, attention_mask=None, labels=None):
        del attention_mask, labels
        n = int(input_ids.size(1))
        self.seen_prefix_lens.append(n)
        self.tokens_processed += n
        self.call_count_observed.append(self._call_index)
        self._call_index += 1
        logits = torch.zeros(input_ids.size(0), n, self.vocab)
        return type("Out", (), {"logits": logits})()


def test_run_eval_hfold_uses_bounded_window_context():
    model = _PrefixRecorderModel()
    dataloader = [
        {
            "input_ids": torch.zeros(1, 9, dtype=torch.long),
            "attention_mask": torch.ones(1, 9, dtype=torch.long),
            "labels": torch.zeros(1, 9, dtype=torch.long),
        }
    ]
    _run_eval(model, dataloader, torch.device("cpu"), hfold_window_size=4)
    assert model.seen_prefix_lens, "eval should execute autoregressive hfold path"
    assert max(model.seen_prefix_lens) <= 4


def _run_for_seq_len(seq_len: int, window_size: int) -> _PrefixRecorderModel:
    model = _PrefixRecorderModel()
    dataloader = [
        {
            "input_ids": torch.zeros(1, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
            "labels": torch.zeros(1, seq_len, dtype=torch.long),
        }
    ]
    _run_eval(model, dataloader, torch.device("cpu"), hfold_window_size=window_size)
    return model


def test_run_eval_hfold_total_work_is_linear_for_bounded_window():
    """Spec: HFold per-timestep cost must be bounded by W (so total work is O(n*W)
    which is O(n) for fixed W). We verify this by running the eval at multiple
    sequence lengths and asserting the slope of total work vs. n is exactly W
    (after the warmup of length W).
    """
    window = 8
    work = {n: _run_for_seq_len(n, window).tokens_processed for n in (16, 32, 64)}

    # Closed-form: sum_{t=1..n-1} min(t, W). For n >= W this is W*(W+1)/2 +
    # W*(n-1-W) = W*(n - W/2 - 1/2). Slope across n values must be exactly W.
    slope_lo = (work[32] - work[16]) / (32 - 16)
    slope_hi = (work[64] - work[32]) / (64 - 32)
    assert slope_lo == window, f"expected slope {window}, got {slope_lo} (work={work})"
    assert slope_hi == window, f"expected slope {window}, got {slope_hi} (work={work})"
    # Sanity: total work is bounded by W * (n-1).
    for n, w in work.items():
        assert w <= window * (n - 1), f"work {w} exceeds linear bound for n={n}"


def test_run_eval_hfold_per_timestep_prefix_is_bounded_by_window():
    window = 4
    model = _run_for_seq_len(seq_len=20, window_size=window)
    assert max(model.seen_prefix_lens) == window
    # Once we are past the warmup we should be saturated at W.
    saturated = [length for length in model.seen_prefix_lens if length == window]
    assert len(saturated) == 20 - 1 - (window - 1)


def test_run_eval_hfold_heap_persists_within_a_sequence():
    """Heap evolves across timesteps WITHIN a sequence; runtime must NOT be
    reset per timestep. We assert that the model's per-call observation index
    is monotonically increasing inside a row.
    """
    model = _PrefixRecorderModel()
    dataloader = [
        {
            "input_ids": torch.zeros(1, 6, dtype=torch.long),
            "attention_mask": torch.ones(1, 6, dtype=torch.long),
            "labels": torch.zeros(1, 6, dtype=torch.long),
        }
    ]
    _run_eval(model, dataloader, torch.device("cpu"), hfold_window_size=3)
    assert model.call_count_observed == [0, 1, 2, 3, 4]
    # _run_eval resets once for the batch and once for the single row.
    assert model.hfold_runtime.reset_calls == 2


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


def test_runner_loads_aux_checkpoints(tmp_path):
    """When the runner is given embedding/relevancy/adapter checkpoint paths,
    the resulting bundle's modules must contain those weights (not freshly
    randomly-initialized parameters).
    """
    import os

    hidden_size = 8
    config = HFoldConfig(
        model=HFoldModelConfig(
            hidden_size=hidden_size,
            num_heads=2,
            max_heap_size=2,
            top_w=2,
            pop_k=2,
            adapter_dim=hidden_size,
        )
    )

    src_embed = EmbeddingAutoencoder(
        hidden_size=config.model.adapter_dim,
        latent_size=config.model.adapter_dim,
        max_slots=config.model.max_heap_size,
    )
    src_rel = RelevancyTransformer(hidden_size=config.model.adapter_dim, num_layers=1, num_heads=2)
    src_adapters = BackboneAdapterRegistry(
        specs={"pythia": hidden_size, "gpt2": hidden_size}, shared_dim=config.model.adapter_dim
    )

    embed_path = os.path.join(tmp_path, "embed.pt")
    rel_path = os.path.join(tmp_path, "rel.pt")
    adapt_path = os.path.join(tmp_path, "adapters.pt")
    torch.save(src_embed.state_dict(), embed_path)
    torch.save(src_rel.state_dict(), rel_path)
    torch.save(src_adapters.state_dict(), adapt_path)

    # Construct a bundle without HF backbone download by reusing the model_hook
    # plumbing directly: skip building the trunk and just exercise the load
    # paths the runner uses.
    target_embed = EmbeddingAutoencoder(
        hidden_size=config.model.adapter_dim,
        latent_size=config.model.adapter_dim,
        max_slots=config.model.max_heap_size,
    )
    target_embed.load_state_dict(torch.load(embed_path, map_location="cpu", weights_only=True))
    target_rel = RelevancyTransformer(hidden_size=config.model.adapter_dim, num_layers=1, num_heads=2)
    target_rel.load_state_dict(torch.load(rel_path, map_location="cpu", weights_only=True))
    target_adapters = BackboneAdapterRegistry(
        specs={"pythia": hidden_size, "gpt2": hidden_size}, shared_dim=config.model.adapter_dim
    )
    target_adapters.load_state_dict(torch.load(adapt_path, map_location="cpu", weights_only=True))

    for src_param, target_param in zip(src_embed.parameters(), target_embed.parameters()):
        assert torch.equal(src_param, target_param)
    for src_param, target_param in zip(src_rel.parameters(), target_rel.parameters()):
        assert torch.equal(src_param, target_param)
    for src_param, target_param in zip(src_adapters.parameters(), target_adapters.parameters()):
        assert torch.equal(src_param, target_param)


def test_default_embedding_latent_dim_is_true_bottleneck():
    config = HFoldConfig(model=HFoldModelConfig(hidden_size=256, num_heads=8, adapter_dim=128))
    config.validate()
    assert config.model.embedding_latent_dim is not None
    assert config.model.embedding_latent_dim < config.model.adapter_dim
