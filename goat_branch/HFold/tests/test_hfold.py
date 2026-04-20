from __future__ import annotations

import unittest

import torch

from hfold import HFoldAttention, HFoldConfig, HFoldTransformerBlock
from hfold.config import HFoldMemoryState


def make_config(**overrides: object) -> HFoldConfig:
    base = dict(
        d_model=16,
        n_heads=4,
        window_size=4,
        heap_size=3,
        top_q=2,
        retrieve_e=2,
        dropout_p=0.0,
        attention_dropout_p=0.0,
    )
    base.update(overrides)
    return HFoldConfig(**base)


class HFoldAttentionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_batched_shapes_and_cache_contract(self) -> None:
        config = make_config()
        module = HFoldAttention(config)
        hidden = torch.randn(2, 6, config.d_model)
        output, cache = module(hidden, use_cache=True)

        self.assertEqual(output.shape, hidden.shape)
        self.assertIsInstance(cache, HFoldMemoryState)
        self.assertEqual(cache.memory_states.shape, (2, config.n_heads, config.heap_size, config.head_dim))
        self.assertEqual(cache.local_keys.shape, (2, config.n_heads, config.window_size, config.head_dim))

    def test_causality_no_future_leakage(self) -> None:
        config = make_config(window_size=3, heap_size=2, top_q=1, retrieve_e=1)
        module = HFoldAttention(config)
        hidden = torch.randn(1, 6, config.d_model)
        altered = hidden.clone()
        altered[:, -1, :] = altered[:, -1, :] + 100.0

        output_a, _ = module(hidden)
        output_b, _ = module(altered)

        self.assertTrue(torch.allclose(output_a[:, :-1, :], output_b[:, :-1, :], atol=1e-5, rtol=1e-5))

    def test_pop_retrieval_removes_entries_from_memory(self) -> None:
        config = make_config(d_model=8, n_heads=1, heap_size=3, retrieve_e=2)
        module = HFoldAttention(config)
        state = HFoldMemoryState.empty(1, config, device=torch.device("cpu"), dtype=torch.float32)
        state.memory_states[0, 0, :, :] = torch.randn(3, config.head_dim)
        state.memory_scores[0, 0, :] = torch.tensor([0.9, 0.5, 0.1])
        state.memory_positions[0, 0, :] = torch.tensor([9, 5, 1])
        state.memory_valid_mask[0, 0, :] = True

        popped = module._pop_memory(state=state, token_valid=torch.tensor([True]))
        updated_state = popped["state"]

        self.assertEqual(popped["counts"][0, 0].item(), 2)
        self.assertEqual(updated_state.memory_valid_mask[0, 0].sum().item(), 1)
        self.assertTrue(torch.equal(popped["positions"][0, 0], torch.tensor([9, 5])))
        self.assertAlmostEqual(updated_state.memory_scores[0, 0, 0].item(), 0.1, places=6)

    def test_heap_bounds_and_removed_counts(self) -> None:
        config = make_config(window_size=5, heap_size=2, top_q=3, retrieve_e=1)
        module = HFoldAttention(config)
        hidden = torch.randn(2, 8, config.d_model)
        _, cache = module(hidden, use_cache=True)
        assert cache is not None

        self.assertTrue((cache.memory_valid_mask.sum(dim=-1) <= config.heap_size).all())
        self.assertTrue((cache.last_retrieved_counts <= config.retrieve_e).all())
        self.assertTrue((cache.last_candidate_counts <= config.top_q).all())
        self.assertTrue((cache.last_removed_counts <= (config.retrieve_e + config.top_q + 1)).all())
        self.assertTrue((cache.local_valid_mask.sum(dim=-1) <= config.window_size).all())

    def test_aged_out_local_tokens_are_tracked(self) -> None:
        config = make_config(d_model=8, n_heads=1, window_size=2, heap_size=2)
        module = HFoldAttention(config)
        state = HFoldMemoryState.empty(1, config, device=torch.device("cpu"), dtype=torch.float32)
        state.local_keys[0, 0, 0, :] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        state.local_keys[0, 0, 1, :] = torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        state.local_values.copy_(state.local_keys)
        state.local_positions[0, :] = torch.tensor([0, 1])
        state.local_valid_mask[0, :] = True

        local = module._advance_local_window(
            state=state,
            current_k=torch.tensor([[[3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]),
            current_v=torch.tensor([[[3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]),
            current_positions=torch.tensor([2]),
            token_valid=torch.tensor([True]),
        )

        self.assertTrue(local["aged_out_valid"][0, 0, 0].item())
        self.assertEqual(local["aged_out_positions"][0, 0].item(), 0)
        self.assertTrue(torch.equal(local["positions"][0], torch.tensor([1, 2])))

    def test_local_attention_uses_previous_k_tokens_only(self) -> None:
        config = make_config(d_model=8, n_heads=1, window_size=2, heap_size=2)
        module = HFoldAttention(config)
        state = HFoldMemoryState.empty(1, config, device=torch.device("cpu"), dtype=torch.float32)
        state.local_keys[0, 0, 0, :] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        state.local_keys[0, 0, 1, :] = torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        state.local_values.copy_(state.local_keys)
        state.local_positions[0, :] = torch.tensor([0, 1])
        state.local_valid_mask[0, :] = True

        local = module._build_local_context(state=state)
        self.assertTrue(torch.equal(local["positions"][0], torch.tensor([0, 1])))
        self.assertTrue(local["valid"][0, 0].all().item())

    def test_top_q_candidates_are_selected_from_local_window(self) -> None:
        config = make_config(d_model=8, n_heads=1, window_size=3, heap_size=2, top_q=1)
        module = HFoldAttention(config)
        local_states = torch.tensor(
            [[[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]]
        )
        local_positions = torch.tensor([[4, 5, 6]], dtype=torch.long)
        local_valid = torch.tensor([[[True, True, True]]])
        local_attn_weights = torch.tensor([[[0.1, 0.2, 0.7]]])

        candidates = module._select_local_candidates(
            local_states=local_states,
            local_positions=local_positions,
            local_valid=local_valid,
            local_attn_weights=local_attn_weights,
            current_positions=torch.tensor([7]),
            token_valid=torch.tensor([True]),
        )

        self.assertTrue(candidates["valid"][0, 0, 0].item())
        self.assertEqual(candidates["positions"][0, 0, 0].item(), 6)

    def test_salience_policy_retains_highest_scoring_entries(self) -> None:
        config = make_config(d_model=8, n_heads=1, heap_size=2, top_q=2, retrieve_e=1)
        module = HFoldAttention(config)

        memory_states = torch.zeros(1, 1, 2, config.head_dim)
        memory_scores = torch.tensor([[[0.3, 0.2]]])
        memory_positions = torch.tensor([[[3, 4]]], dtype=torch.long)
        memory_fold_counts = torch.zeros(1, 1, 2, dtype=torch.long)
        memory_valid_mask = torch.tensor([[[True, True]]])

        candidate_states = torch.randn(1, 1, 2, config.head_dim)
        candidate_scores = torch.tensor([[[0.8, 0.1]]])
        candidate_positions = torch.tensor([[[7, 8]]], dtype=torch.long)
        candidate_valid_mask = torch.tensor([[[True, True]]])

        state = HFoldMemoryState(
            memory_states=memory_states,
            memory_scores=memory_scores,
            memory_positions=memory_positions,
            memory_fold_counts=memory_fold_counts,
            memory_valid_mask=memory_valid_mask,
            local_keys=torch.zeros(1, 1, config.window_size, config.head_dim),
            local_values=torch.zeros(1, 1, config.window_size, config.head_dim),
            local_positions=torch.full((1, config.window_size), -1, dtype=torch.long),
            local_valid_mask=torch.zeros(1, config.window_size, dtype=torch.bool),
            next_positions=torch.zeros(1, dtype=torch.long),
        )
        updated_state, overflow = module._insert_candidates_into_memory(
            state=state,
            candidate_states=candidate_states,
            candidate_scores=candidate_scores,
            candidate_positions=candidate_positions,
            candidate_valid_mask=candidate_valid_mask,
        )

        retained_scores = updated_state.memory_scores[0, 0, updated_state.memory_valid_mask[0, 0]].tolist()
        self.assertEqual(len(retained_scores), 2)
        self.assertGreaterEqual(min(retained_scores), 0.3)
        self.assertEqual(overflow["counts"][0, 0].item(), 2)

    def test_all_retained_nodes_fold_removed_set(self) -> None:
        config = make_config(d_model=8, n_heads=1, heap_size=2, top_q=2, retrieve_e=1)
        module = HFoldAttention(config)

        memory_states = torch.randn(1, 1, 2, config.head_dim)
        memory_scores = torch.tensor([[[0.7, 0.4]]])
        memory_positions = torch.tensor([[[4, 5]]], dtype=torch.long)
        memory_fold_counts = torch.zeros(1, 1, 2, dtype=torch.long)
        memory_valid_mask = torch.tensor([[[True, True]]])

        candidate_states = torch.randn(1, 1, 2, config.head_dim)
        candidate_scores = torch.tensor([[[0.9, 0.6]]])
        candidate_positions = torch.tensor([[[6, 7]]], dtype=torch.long)
        candidate_valid_mask = torch.tensor([[[True, True]]])

        popped_states = torch.randn(1, 1, 1, config.head_dim)
        popped_scores = torch.tensor([[[0.8]]])
        popped_positions = torch.tensor([[[2]]], dtype=torch.long)
        popped_valid_mask = torch.tensor([[[True]]])

        state = HFoldMemoryState(
            memory_states=memory_states,
            memory_scores=memory_scores,
            memory_positions=memory_positions,
            memory_fold_counts=memory_fold_counts,
            memory_valid_mask=memory_valid_mask,
            local_keys=torch.zeros(1, 1, config.window_size, config.head_dim),
            local_values=torch.zeros(1, 1, config.window_size, config.head_dim),
            local_positions=torch.full((1, config.window_size), -1, dtype=torch.long),
            local_valid_mask=torch.zeros(1, config.window_size, dtype=torch.bool),
            next_positions=torch.zeros(1, dtype=torch.long),
        )
        inserted_state, overflow = module._insert_candidates_into_memory(
            state=state,
            candidate_states=candidate_states,
            candidate_scores=candidate_scores,
            candidate_positions=candidate_positions,
            candidate_valid_mask=candidate_valid_mask,
        )
        removed_states = torch.cat([popped_states, overflow["states"]], dim=2)
        removed_scores = torch.cat([popped_scores, overflow["scores"]], dim=2)
        removed_positions = torch.cat([popped_positions, overflow["positions"]], dim=2)
        removed_valid = torch.cat([popped_valid_mask, overflow["valid"]], dim=2)
        folded_state = module._fold_removed_memory_state(
            state=inserted_state,
            removed_states=removed_states,
            removed_scores=removed_scores,
            removed_positions=removed_positions,
            removed_valid=removed_valid,
        )

        valid_fold_counts = folded_state.memory_fold_counts[0, 0, folded_state.memory_valid_mask[0, 0]]
        self.assertEqual(valid_fold_counts.numel(), 2)
        self.assertTrue((valid_fold_counts > 0).all())
        self.assertEqual(removed_valid.sum().item(), 3)

    def test_retrieve_zero_matches_heap_disabled(self) -> None:
        config_a = make_config(retrieve_e=0, heap_size=3)
        config_b = make_config(retrieve_e=0, heap_size=0)
        module_a = HFoldAttention(config_a)
        module_b = HFoldAttention(config_b)
        module_b.load_state_dict(module_a.state_dict(), strict=False)

        hidden = torch.randn(2, 6, config_a.d_model)
        output_a, _ = module_a(hidden)
        output_b, _ = module_b(hidden)
        self.assertTrue(torch.allclose(output_a, output_b, atol=1e-5, rtol=1e-5))

    def test_batched_and_incremental_decode_match(self) -> None:
        config = make_config(window_size=4, heap_size=3, top_q=2, retrieve_e=2)
        module = HFoldAttention(config)
        hidden = torch.randn(2, 7, config.d_model)

        full_output, _ = module(hidden)

        cache = None
        step_outputs = []
        for index in range(hidden.size(1)):
            step_output, cache = module(hidden[:, index : index + 1, :], cache=cache, use_cache=True)
            step_outputs.append(step_output)

        incremental_output = torch.cat(step_outputs, dim=1)
        self.assertTrue(torch.allclose(full_output, incremental_output, atol=1e-5, rtol=1e-5))

    def test_reference_block_trains_without_nans(self) -> None:
        config = make_config(d_model=12, n_heads=3, window_size=3, heap_size=2)
        block = HFoldTransformerBlock(config)
        optimizer = torch.optim.AdamW(block.parameters(), lr=1e-3)

        for _ in range(3):
            hidden = torch.randn(2, 5, config.d_model)
            target = torch.randn(2, 5, config.d_model)
            output, _ = block(hidden)
            loss = torch.nn.functional.mse_loss(output, target)
            self.assertFalse(torch.isnan(loss).item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    unittest.main()
