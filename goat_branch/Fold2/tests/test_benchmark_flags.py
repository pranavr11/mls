import pytest


def test_evaluate_can_disable_throughput_and_memory_profiling():
    torch = pytest.importorskip("torch")

    from hfold_pipeline.config import TrainingConfig
    from hfold_pipeline.training.trainer import FixedLengthCausalLMCollator, evaluate

    class TinyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(16, 8)
            self.lm_head = torch.nn.Linear(8, 16)

        def forward(self, input_ids, attention_mask=None, labels=None):
            del attention_mask
            hidden = self.embed(input_ids)
            logits = self.lm_head(hidden)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
            )
            return type("TinyOutput", (), {"loss": loss, "logits": logits})()

    model = TinyLM()
    batch = {
        "input_ids": [1, 2, 3, 4],
        "attention_mask": [1, 1, 1, 1],
        "labels": [1, 2, 3, 4],
    }
    collator = FixedLengthCausalLMCollator()
    dataloader = [collator([batch])]

    metrics = evaluate(
        model=model,
        dataloader=dataloader,
        device=torch.device("cpu"),
        training_config=TrainingConfig(),
        profile_throughput=False,
        profile_memory=False,
    )

    assert metrics["tokens_per_second"] is None
    assert metrics["peak_memory_mb"] is None

