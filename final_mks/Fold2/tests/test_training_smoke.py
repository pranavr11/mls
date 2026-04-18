from pathlib import Path

import pytest


def test_train_smoke_runs_for_full_and_sliding(monkeypatch, tmp_path: Path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from hfold_pipeline.config import ExperimentConfig
    from hfold_pipeline.training import trainer as trainer_module

    class TinyTokenizer:
        pad_token = "<pad>"
        eos_token = "</s>"

        def save_pretrained(self, path):
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    class TinyConfig:
        def save_pretrained(self, path):
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            (path / "config.json").write_text("{}", encoding="utf-8")

    class TinyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(32, 16)
            self.lm_head = torch.nn.Linear(16, 32)
            self.config = TinyConfig()

        def forward(self, input_ids, attention_mask=None, labels=None):
            del attention_mask
            hidden = self.embed(input_ids)
            logits = self.lm_head(hidden)
            loss = None
            if labels is not None:
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1),
                )
            return type("TinyOutput", (), {"loss": loss, "logits": logits})()

        def gradient_checkpointing_enable(self):
            return None

    def fake_model_and_tokenizer(_config):
        return TinyLM(), TinyTokenizer()

    tiny_split = [
        {
            "input_ids": [1, 2, 3, 4],
            "attention_mask": [1, 1, 1, 1],
            "labels": [1, 2, 3, 4],
        },
        {
            "input_ids": [4, 3, 2, 1],
            "attention_mask": [1, 1, 1, 1],
            "labels": [4, 3, 2, 1],
        },
    ]

    def fake_dataset(_tokenizer, _data_config):
        return {"train": tiny_split, "validation": tiny_split, "test": tiny_split}

    def fake_save_checkpoint(**kwargs):
        path = Path(kwargs["checkpoint_dir"])
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(trainer_module, "load_pythia_model_and_tokenizer", fake_model_and_tokenizer)
    monkeypatch.setattr(trainer_module, "load_or_prepare_pg19", fake_dataset)
    monkeypatch.setattr(trainer_module, "save_checkpoint", fake_save_checkpoint)
    monkeypatch.setattr(trainer_module, "cleanup_old_checkpoints", lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer_module, "find_latest_checkpoint", lambda *_args, **_kwargs: None)

    for attention_type in ("full", "sliding_window"):
        config = ExperimentConfig()
        config.attention.attention_type = attention_type
        config.training.output_dir = str(tmp_path / attention_type)
        config.training.max_steps = 1
        config.training.num_train_epochs = 1.0
        config.training.per_device_batch_size = 1
        config.training.gradient_accumulation_steps = 1
        config.training.eval_interval = 100
        config.training.save_interval = 100
        config.training.log_interval = 1
        config.training.dataloader_num_workers = 0
        config.runtime.device = "cpu"

        result = trainer_module.train(config)
        assert "loss" in result
        assert result["num_batches"] >= 1

