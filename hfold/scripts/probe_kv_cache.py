from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from hfold._debug_log import debug_log
from hfold.config.schema import HFoldConfig, HFoldModelConfig
from hfold.inference.hfold_runtime import HFoldRuntime
from hfold.inference.model_hook import wrap_pythia_with_hfold
from hfold.models.adapters import BackboneAdapterRegistry
from hfold.models.embedding_autoencoder import EmbeddingAutoencoder
from hfold.models.relevancy_transformer import RelevancyTransformer


@dataclass
class _TrunkOutput:
    last_hidden_state: torch.Tensor
    attentions: tuple[torch.Tensor, ...]
    past_key_values: tuple[torch.Tensor, ...] | None = None


class _CacheAwareTrunk(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int = 32) -> None:
        super().__init__()
        self.embed_in = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.call_records: list[dict[str, object]] = []

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        output_attentions=False,
        return_dict=True,
        use_cache=False,
        past_key_values=None,
        **_kwargs,
    ):
        del return_dict
        h = inputs_embeds if inputs_embeds is not None else self.embed_in(input_ids)
        b, s, _ = h.shape
        self.call_records.append(
            {
                "seq_len": int(s),
                "use_cache": bool(use_cache),
                "has_past": past_key_values is not None,
                "mask_shape": None if attention_mask is None else tuple(int(x) for x in attention_mask.shape),
            }
        )
        h = self.proj(h)
        attns = (torch.softmax(torch.zeros(b, 1, s, s, device=h.device), dim=-1),) if output_attentions else tuple()
        # dummy cache payload
        pkv = (torch.zeros(1, s, h.size(-1)),) if use_cache else None
        return _TrunkOutput(last_hidden_state=h, attentions=attns, past_key_values=pkv)


class _Model(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.gpt_neox = _CacheAwareTrunk(hidden_size)


def main() -> None:
    torch.manual_seed(0)
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
    runtime = HFoldRuntime(config)
    runtime.attach_adapters(
        BackboneAdapterRegistry(specs={"pythia": hidden_size}, shared_dim=hidden_size),
        "pythia",
    )
    model = _Model(hidden_size)
    embed = EmbeddingAutoencoder(hidden_size=hidden_size, latent_size=hidden_size, max_slots=config.model.max_heap_size)
    rel = RelevancyTransformer(hidden_size=hidden_size, num_layers=1, num_heads=2)
    wrap_pythia_with_hfold(model, runtime, embed, rel)

    input_ids = torch.randint(0, 16, (1, 3))
    # step 0
    out0 = model.gpt_neox(input_ids=input_ids, use_cache=True)
    # step t with past
    _ = model.gpt_neox(input_ids=input_ids[:, :1], use_cache=True, past_key_values=out0.past_key_values)

    # region agent log
    debug_log(
        hypothesis_id="H12",
        location="hfold/scripts/probe_kv_cache.py:main",
        message="cache behavior before fix",
        data={
            "num_calls": len(model.gpt_neox.call_records),
            "call_records": model.gpt_neox.call_records,
        },
    )
    # endregion


if __name__ == "__main__":
    main()
