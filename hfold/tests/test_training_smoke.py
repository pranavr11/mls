from hfold.config.schema import HFoldConfig, HFoldModelConfig, HFoldTrainingConfig
from hfold.data.hidden_state_dataset import SyntheticHiddenStateDataset
from hfold.training.train_embedding import train_embedding_model
from hfold.training.train_relevancy import train_relevancy_model


def test_embedding_and_relevancy_training_smoke():
    config = HFoldConfig(
        model=HFoldModelConfig(hidden_size=16, num_heads=4, max_heap_size=4, adapter_dim=16),
        training=HFoldTrainingConfig(num_epochs=1, max_steps=2, batch_size=2),
    )
    dataset = SyntheticHiddenStateDataset(size=8, backbone="pythia", hidden_size=16, max_heap_size=4)
    emb = train_embedding_model(config=config, dataset=dataset, backbone_dims={"pythia": 16, "gpt2": 16})
    rel = train_relevancy_model(config=config, dataset=dataset, embedding_model=emb.model, adapters=emb.adapters)
    assert emb.final_loss >= 0.0
    assert rel.final_loss >= 0.0
