import copy
import enum
import importlib
import importlib.machinery
import os
import sys
import types

import pytest


def _install_torchvision_stub():
    torchvision = types.ModuleType("torchvision")
    torchvision.__spec__ = importlib.machinery.ModuleSpec("torchvision", loader=None)
    torchvision.__path__ = []
    torchvision.__version__ = "0.0"

    class InterpolationMode(enum.Enum):
        NEAREST = 0
        NEAREST_EXACT = 1
        BILINEAR = 2
        BICUBIC = 3
        BOX = 4
        HAMMING = 5
        LANCZOS = 6

    transforms = types.ModuleType("torchvision.transforms")
    transforms.__spec__ = importlib.machinery.ModuleSpec("torchvision.transforms", loader=None)
    transforms.InterpolationMode = InterpolationMode

    functional = types.ModuleType("torchvision.transforms.functional")
    functional.__spec__ = importlib.machinery.ModuleSpec(
        "torchvision.transforms.functional", loader=None
    )

    torchvision.transforms = transforms
    sys.modules["torchvision"] = torchvision
    sys.modules["torchvision.transforms"] = transforms
    sys.modules["torchvision.transforms.functional"] = functional

    for name in ("_meta_registrations", "datasets", "io", "models", "ops", "utils"):
        module = types.ModuleType(f"torchvision.{name}")
        module.__spec__ = importlib.machinery.ModuleSpec(f"torchvision.{name}", loader=None)
        setattr(torchvision, name, module)
        sys.modules[f"torchvision.{name}"] = module


def _clear_modules(prefix: str):
    for name in list(sys.modules):
        if name == prefix or name.startswith(f"{prefix}."):
            sys.modules.pop(name, None)


def _import_gptneox_classes():
    try:
        config_module = importlib.import_module(
            "transformers.models.gpt_neox.configuration_gpt_neox"
        )
        modeling_module = importlib.import_module(
            "transformers.models.gpt_neox.modeling_gpt_neox"
        )
        return config_module.GPTNeoXConfig, modeling_module.GPTNeoXForCausalLM
    except ModuleNotFoundError:
        pytest.skip("transformers is not installed")
    except Exception as exc:
        message = repr(exc)
        if "torchvision" not in message and "nms" not in message:
            raise

    _clear_modules("transformers")
    _clear_modules("torchvision")
    _install_torchvision_stub()

    try:
        config_module = importlib.import_module(
            "transformers.models.gpt_neox.configuration_gpt_neox"
        )
        modeling_module = importlib.import_module(
            "transformers.models.gpt_neox.modeling_gpt_neox"
        )
        return config_module.GPTNeoXConfig, modeling_module.GPTNeoXForCausalLM
    except Exception as exc:  # pragma: no cover - environment-dependent fallback
        pytest.skip(f"Unable to import GPT-NeoX classes in this environment: {exc}")


def _import_transformers_module():
    try:
        return importlib.import_module("transformers")
    except ModuleNotFoundError:
        pytest.skip("transformers is not installed")
    except Exception as exc:
        message = repr(exc)
        if "torchvision" not in message and "nms" not in message:
            raise

    _clear_modules("transformers")
    _clear_modules("torchvision")
    _install_torchvision_stub()
    try:
        return importlib.import_module("transformers")
    except Exception as exc:  # pragma: no cover - environment-dependent fallback
        pytest.skip(f"Unable to import transformers in this environment: {exc}")


def test_full_attention_strategy_matches_unpatched_gptneox():
    torch = pytest.importorskip("torch")
    GPTNeoXConfig, GPTNeoXForCausalLM = _import_gptneox_classes()

    from hfold_pipeline.config import AttentionConfig
    from hfold_pipeline.modeling.pythia import patch_model_attention

    config = GPTNeoXConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        use_cache=False,
    )
    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = "eager"

    baseline = GPTNeoXForCausalLM(config).eval()
    patched = copy.deepcopy(baseline).eval()
    patch_model_attention(patched, AttentionConfig(attention_type="full"))

    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        baseline_outputs = baseline(input_ids=input_ids, attention_mask=attention_mask)
        patched_outputs = patched(input_ids=input_ids, attention_mask=attention_mask)

    torch.testing.assert_close(
        patched_outputs.logits,
        baseline_outputs.logits,
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.skipif(
    os.environ.get("HFOLD_RUN_PYTHIA_PARITY") != "1",
    reason="Set HFOLD_RUN_PYTHIA_PARITY=1 to run the real Pythia parity test.",
)
def test_full_attention_strategy_matches_pretrained_pythia():
    torch = pytest.importorskip("torch")
    transformers = _import_transformers_module()

    from hfold_pipeline.config import AttentionConfig
    from hfold_pipeline.modeling.pythia import patch_model_attention

    baseline = transformers.AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-160m",
        trust_remote_code=False,
    ).eval()
    patched = transformers.AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-160m",
        trust_remote_code=False,
    ).eval()
    patch_model_attention(patched, AttentionConfig(attention_type="full"))

    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        baseline_outputs = baseline(input_ids=input_ids, attention_mask=attention_mask)
        patched_outputs = patched(input_ids=input_ids, attention_mask=attention_mask)

    torch.testing.assert_close(
        patched_outputs.logits,
        baseline_outputs.logits,
        rtol=1e-5,
        atol=1e-5,
    )
