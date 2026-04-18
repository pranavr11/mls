from pathlib import Path

from hfold_pipeline.config import load_experiment_config


def test_load_experiment_config(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  model_name: EleutherAI/pythia-160m
attention:
  attention_type: sliding_window
  window_size: 256
data:
  block_size: 1024
training:
  output_dir: artifacts/test
runtime:
  seed: 7
benchmark:
  sequence_lengths: [128, 256]
""",
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)

    assert config.model.model_name == "EleutherAI/pythia-160m"
    assert config.attention.attention_type == "sliding_window"
    assert config.attention.window_size == 256
    assert config.runtime.seed == 7
    assert config.benchmark.sequence_lengths == [128, 256]


def test_attention_aliases_are_normalized(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
attention:
  attention_type: self_attention
""",
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)
    assert config.attention.attention_type == "full"


def test_sliding_attention_alias_is_normalized(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
attention:
  attention_type: sliding_attention
""",
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)
    assert config.attention.attention_type == "sliding_window"


def test_null_hfold_backend_is_allowed_for_native_backend(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
attention:
  attention_type: hfold
  hfold_backend:
""",
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)
    assert config.attention.hfold_backend is None


def test_placeholder_hfold_backend_now_falls_back_to_native_backend(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
attention:
  attention_type: hfold
  hfold_backend: your_package.your_module:build_hfold_backend
""",
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)
    assert config.attention.attention_type == "hfold"
    assert config.attention.hfold_backend is None
