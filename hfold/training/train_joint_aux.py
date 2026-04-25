from __future__ import annotations

from dataclasses import dataclass

from ..config.schema import HFoldConfig
from ..models.adapters import BackboneAdapterRegistry
from ..models.embedding_autoencoder import EmbeddingAutoencoder
from ..models.relevancy_transformer import RelevancyTransformer
from .train_embedding import EmbeddingTrainingArtifacts, train_embedding_model
from .train_relevancy import RelevancyTrainingArtifacts, train_relevancy_model


@dataclass
class JointAuxTrainingArtifacts:
    embedding: EmbeddingAutoencoder
    relevancy: RelevancyTransformer
    adapters: BackboneAdapterRegistry
    embedding_loss: float
    relevancy_loss: float


def train_joint_aux_models(
    *,
    config: HFoldConfig,
    dataset,
    backbone_dims: dict[str, int],
) -> JointAuxTrainingArtifacts:
    emb_artifacts: EmbeddingTrainingArtifacts = train_embedding_model(
        config=config,
        dataset=dataset,
        backbone_dims=backbone_dims,
    )
    rel_artifacts: RelevancyTrainingArtifacts = train_relevancy_model(
        config=config,
        dataset=dataset,
        embedding_model=emb_artifacts.model,
        adapters=emb_artifacts.adapters,
    )
    return JointAuxTrainingArtifacts(
        embedding=emb_artifacts.model,
        relevancy=rel_artifacts.model,
        adapters=emb_artifacts.adapters,
        embedding_loss=emb_artifacts.final_loss,
        relevancy_loss=rel_artifacts.final_loss,
    )
