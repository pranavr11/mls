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
        latent_size=config.model.adapter_dim,
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
            evicted_vectors = batch["evicted_vectors"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            encoded = []
            for row, backbone in enumerate(backbones):
                encoded.append(adapters.encode(backbone, evicted_vectors[row]))
            encoded_batch = torch.stack(encoded, dim=0)
            reconstructed, _ = model(encoded_batch, padding_mask=padding_mask)
            decoded = []
            for row, backbone in enumerate(backbones):
                decoded.append(adapters.decode(backbone, reconstructed[row]))
            decoded_batch = torch.stack(decoded, dim=0)
            loss = cosine_reconstruction_loss(decoded_batch, evicted_vectors, mask=padding_mask)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(adapters.parameters()), config.training.gradient_clip_norm)
            optimizer.step()
            final_loss = float(loss.item())
            step += 1
            if config.training.max_steps is not None and step >= config.training.max_steps:
                return EmbeddingTrainingArtifacts(model=model, adapters=adapters, final_loss=final_loss)

    return EmbeddingTrainingArtifacts(model=model, adapters=adapters, final_loss=final_loss)
