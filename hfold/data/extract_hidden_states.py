"""Extract real hidden states + teacher attention scores from a fine-tuned
sliding-window backbone.

For each chunk of tokens, we run the model with output_attentions=True and
output_hidden_states=True. From the **last** layer:
  * `H ∈ [seq, hidden]` — hidden states.
  * `A ∈ [heads, seq, seq]` — attention probabilities.

For each anchor query position `q` we sample, we sort previous-token keys by
mean-over-heads attention-from-q. The training tuple is:

  - heap_vectors:    H[top-S keys]                shape [S, hidden]
  - evicted_vectors: H[next-S keys after top-S]   shape [S, hidden] (those
                                                  the heap would push out)
  - teacher_scores:  softmaxed attention probs from q over the heap_vectors
                     shape [S], sums to 1.

This file produces shards consumable by `HiddenStateShardDataset` with no
placeholder values anywhere.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F


@dataclass
class ExtractionConfig:
    backbone: str
    chunk_len: int = 512
    max_heap_size: int = 16
    num_anchors_per_chunk: int = 4
    min_anchor_position: int | None = None
    seed: int = 42


def _resolve_min_anchor(config: ExtractionConfig) -> int:
    """Anchor must have ≥ 2*S previous keys so we can pick S heap + S evicted."""
    if config.min_anchor_position is not None:
        return int(config.min_anchor_position)
    return 2 * config.max_heap_size + 1


def extract_one_chunk(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    config: ExtractionConfig,
    generator: torch.Generator,
) -> list[dict]:
    """Return up to `num_anchors_per_chunk` real training tuples for one chunk.

    `input_ids` shape: [1, chunk_len].
    """
    device = input_ids.device
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    last_hidden = outputs.hidden_states[-1][0]  # [seq, hidden]
    last_attn = outputs.attentions[-1][0]  # [heads, seq, seq]
    seq_len = last_hidden.size(0)
    min_anchor = _resolve_min_anchor(config)
    if seq_len <= min_anchor:
        return []

    candidate_anchors = torch.arange(min_anchor, seq_len, device=device)
    if candidate_anchors.numel() == 0:
        return []
    num_to_sample = min(config.num_anchors_per_chunk, int(candidate_anchors.numel()))
    perm = torch.randperm(candidate_anchors.numel(), generator=generator).to(device)
    selected = candidate_anchors[perm[:num_to_sample]]

    samples: list[dict] = []
    s = config.max_heap_size
    for anchor in selected.tolist():
        # Mean over heads of attention from `anchor` to all keys.
        attn_q = last_attn[:, anchor, :].mean(dim=0)
        prev_attn = attn_q[:anchor]
        if prev_attn.numel() < 2 * s:
            continue
        sorted_scores, sorted_idx = torch.sort(prev_attn, descending=True)
        heap_idx = sorted_idx[:s]
        evicted_idx = sorted_idx[s : 2 * s]
        heap_vectors = last_hidden[heap_idx].detach().to("cpu")
        evicted_vectors = last_hidden[evicted_idx].detach().to("cpu")
        teacher_raw = sorted_scores[:s].detach().to("cpu")
        teacher_scores = teacher_raw / teacher_raw.sum().clamp_min(1e-9)
        samples.append(
            {
                "backbone": config.backbone,
                "heap_vectors": heap_vectors.contiguous(),
                "evicted_vectors": evicted_vectors.contiguous(),
                "teacher_scores": teacher_scores.contiguous(),
            }
        )
    return samples


def extract_to_shards(
    *,
    model: torch.nn.Module,
    dataloader: Iterable[dict],
    output_dir: str,
    config: ExtractionConfig,
    samples_per_shard: int = 256,
    max_chunks: int | None = None,
) -> int:
    """Run the model over `dataloader`, write `.pt` shards under `output_dir`.

    Each shard is a list of training-tuple dicts. Returns total tuples written.
    """
    os.makedirs(output_dir, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)

    buffer: list[dict] = []
    shard_index = 0
    total = 0
    chunks_seen = 0
    for batch in dataloader:
        if max_chunks is not None and chunks_seen >= max_chunks:
            break
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        if input_ids.dim() != 2:
            raise ValueError(f"Expected input_ids [batch, seq]; got {tuple(input_ids.shape)}")
        for row in range(input_ids.size(0)):
            if max_chunks is not None and chunks_seen >= max_chunks:
                break
            row_ids = input_ids[row : row + 1]
            row_mask = None if attention_mask is None else attention_mask[row : row + 1]
            samples = extract_one_chunk(
                model=model,
                input_ids=row_ids,
                attention_mask=row_mask,
                config=config,
                generator=generator,
            )
            buffer.extend(samples)
            chunks_seen += 1
            while len(buffer) >= samples_per_shard:
                shard_path = os.path.join(output_dir, f"shard_{shard_index:04d}.pt")
                torch.save(buffer[:samples_per_shard], shard_path)
                total += samples_per_shard
                buffer = buffer[samples_per_shard:]
                shard_index += 1
    if buffer:
        shard_path = os.path.join(output_dir, f"shard_{shard_index:04d}.pt")
        torch.save(buffer, shard_path)
        total += len(buffer)
    return total
