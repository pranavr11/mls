from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from ..config.schema import HFoldConfig
from ..data.collate import collate_hfold_samples
from ..models.adapters import BackboneAdapterRegistry
from ..models.embedding_autoencoder import EmbeddingAutoencoder
from .losses import cosine_reconstruction_loss


@dataclass
class EmbeddingTrainingArtifacts:
    model: EmbeddingAutoencoder
    adapters: BackboneAdapterRegistry
    final_loss: float


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def train_embedding_model(
    *,
    config: HFoldConfig,
    dataset,
    backbone_dims: dict[str, int],
) -> EmbeddingTrainingArtifacts:
    config.validate()
    torch.manual_seed(config.training.seed)
    device = _resolve_device(config.training.device)
    loader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True, collate_fn=collate_hfold_samples)

    adapters = BackboneAdapterRegistry(specs=backbone_dims, shared_dim=config.model.adapter_dim).to(device)
    model = EmbeddingAutoencoder(
        hidden_size=config.model.adapter_dim,
        latent_size=int(config.model.embedding_latent_dim),
        max_slots=config.model.max_heap_size,
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(adapters.parameters()),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    final_loss = 0.0
    step = 0
    for _ in range(config.training.num_epochs):
        for batch in loader:
            backbones = batch["backbones"]
            evicted_list = [t.to(device) for t in batch["evicted_vectors"]]
            padding_mask = batch["padding_mask"].to(device)
            # Encode each per-sample evicted block through its backbone adapter
            # into shared latent dim; only stack after that so heterogeneous
            # raw hidden sizes (Pythia vs GPT-2) are supported.
            encoded = [adapters.encode(b, vec) for b, vec in zip(backbones, evicted_list)]
            encoded_batch = torch.stack(encoded, dim=0)
            reconstructed, _ = model(encoded_batch, padding_mask=padding_mask)
            decoded = [adapters.decode(b, reconstructed[i]) for i, b in enumerate(backbones)]
            # Per-sample reconstruction loss in original backbone space.
            losses = [
                cosine_reconstruction_loss(
                    decoded[i].unsqueeze(0),
                    evicted_list[i].unsqueeze(0),
                    mask=padding_mask[i : i + 1],
                )
                for i in range(len(backbones))
            ]
            loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(adapters.parameters()), config.training.gradient_clip_norm)
            optimizer.step()
            final_loss = float(loss.item())
            step += 1
            if config.training.max_steps is not None and step >= config.training.max_steps:
                return EmbeddingTrainingArtifacts(model=model, adapters=adapters, final_loss=final_loss)

    return EmbeddingTrainingArtifacts(model=model, adapters=adapters, final_loss=final_loss)
