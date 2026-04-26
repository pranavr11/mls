from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config.schema import HFoldConfig
from ..data.collate import collate_hfold_samples
from ..models.adapters import BackboneAdapterRegistry
from ..models.embedding_autoencoder import EmbeddingAutoencoder
from ..models.relevancy_transformer import RelevancyTransformer
from .losses import ranking_loss
from .train_embedding import _resolve_device


@dataclass
class RelevancyTrainingArtifacts:
    model: RelevancyTransformer
    final_loss: float


def train_relevancy_model(
    *,
    config: HFoldConfig,
    dataset,
    embedding_model: EmbeddingAutoencoder,
    adapters: BackboneAdapterRegistry,
) -> RelevancyTrainingArtifacts:
    config.validate()
    torch.manual_seed(config.training.seed)
    device = _resolve_device(config.training.device)
    loader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True, collate_fn=collate_hfold_samples)
    model = RelevancyTransformer(hidden_size=config.model.adapter_dim).to(device)
    embedding_model = embedding_model.to(device).eval()
    adapters = adapters.to(device).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)

    final_loss = 0.0
    step = 0
    for _ in range(config.training.num_epochs):
        for batch in loader:
            backbones = batch["backbones"]
            evicted_list = [t.to(device) for t in batch["evicted_vectors"]]
            heap_list = [t.to(device) for t in batch["heap_vectors"]]
            teacher_scores = batch["teacher_scores"].to(device)
            padding_mask = batch["padding_mask"].to(device)

            encoded_evicted = [adapters.encode(b, vec) for b, vec in zip(backbones, evicted_list)]
            encoded_heap = [adapters.encode(b, vec) for b, vec in zip(backbones, heap_list)]
            encoded_evicted_batch = torch.stack(encoded_evicted, dim=0)
            encoded_heap_batch = torch.stack(encoded_heap, dim=0)
            with torch.no_grad():
                summary = embedding_model.encode_summary(encoded_evicted_batch, padding_mask=padding_mask)
            pred_scores = model(summary, encoded_heap_batch)
            target_distribution = teacher_scores / teacher_scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            pred_log_distribution = F.log_softmax(pred_scores, dim=-1)
            pred_distribution = pred_log_distribution.exp()
            kl = F.kl_div(pred_log_distribution, target_distribution, reduction="batchmean")
            mse = F.mse_loss(pred_distribution, target_distribution)
            rank = ranking_loss(pred_scores, teacher_scores)
            loss = kl + config.training.mse_loss_weight * mse + config.training.ranking_loss_weight * rank
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
            optimizer.step()
            final_loss = float(loss.item())
            step += 1
            if config.training.max_steps is not None and step >= config.training.max_steps:
                return RelevancyTrainingArtifacts(model=model, final_loss=final_loss)

    return RelevancyTrainingArtifacts(model=model, final_loss=final_loss)
