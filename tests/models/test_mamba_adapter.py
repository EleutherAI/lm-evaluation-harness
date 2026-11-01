"""Offline tests for the Mamba adapter's model-loading and generation contract."""

import sys
from types import ModuleType
from unittest.mock import Mock, patch

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("accelerate")

from lm_eval.models.huggingface import HFLM
from lm_eval.models.mamba_lm import MambaLMWrapper


@pytest.mark.parametrize(
    "pretrained,is_hf,expected_hf",
    [
        ("state-spaces/mamba-130m", False, False),
        ("state-spaces/mamba-130m-hf", False, True),
        ("local-model", True, True),
    ],
)
def test_constructor_selects_backend_and_default_tokenizer(
    pretrained, is_hf, expected_hf
):
    with patch.object(HFLM, "__init__", return_value=None) as initialize:
        model = MambaLMWrapper(pretrained=pretrained, is_hf=is_hf)

    assert model.is_hf is expected_hf
    initialize.assert_called_once_with(
        pretrained=pretrained,
        backend="causal",
        tokenizer="EleutherAI/gpt-neox-20b",
        max_length=2048,
    )


def test_constructor_forwards_explicit_options():
    with patch.object(HFLM, "__init__", return_value=None) as initialize:
        MambaLMWrapper(
            pretrained="local-model",
            backend="causal",
            tokenizer="local-tokenizer",
            max_length=1024,
            batch_size=4,
            device="cpu",
            dtype="float32",
        )

    initialize.assert_called_once_with(
        pretrained="local-model",
        backend="causal",
        tokenizer="local-tokenizer",
        max_length=1024,
        batch_size=4,
        device="cpu",
        dtype="float32",
    )


def test_constructor_rejects_noncausal_backend_before_loading():
    with (
        patch.object(HFLM, "__init__", return_value=None) as initialize,
        pytest.raises(AssertionError),
    ):
        MambaLMWrapper(pretrained="local-model", backend="seq2seq")

    initialize.assert_not_called()


@pytest.fixture
def native_mamba(monkeypatch):
    """Replace only the optional CUDA package at the model-loading boundary."""
    modules = {
        name: ModuleType(name)
        for name in (
            "mamba_ssm",
            "mamba_ssm.utils",
            "mamba_ssm.utils.hf",
            "mamba_ssm.models",
            "mamba_ssm.models.mixer_seq_simple",
        )
    }
    config_loader = Mock(return_value={"d_model": 16, "n_layer": 2})
    model_class = Mock()
    modules["mamba_ssm.utils.hf"].load_config_hf = config_loader
    modules["mamba_ssm.models.mixer_seq_simple"].MambaLMHeadModel = model_class
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return config_loader, model_class


def test_native_configuration_loading(native_mamba):
    config_loader, _ = native_mamba
    model = object.__new__(MambaLMWrapper)
    model.is_hf = False

    model._get_config("local-mamba")

    config_loader.assert_called_once_with("local-mamba")
    assert model.config == {"d_model": 16, "n_layer": 2}


@pytest.mark.parametrize(
    "dtype,expected_dtype",
    [
        ("auto", torch.float16),
        ("float32", torch.float32),
        ("bfloat16", torch.bfloat16),
        (torch.float64, torch.float64),
    ],
)
def test_native_model_loading_passes_device_and_dtype(
    native_mamba, dtype, expected_dtype
):
    _, model_class = native_mamba
    model = object.__new__(MambaLMWrapper)
    model.is_hf = False
    model._device = torch.device("cpu")

    model._create_model("local-mamba", dtype=dtype)

    model_class.from_pretrained.assert_called_once_with(
        "local-mamba", device=torch.device("cpu"), dtype=expected_dtype
    )
    assert model.model is model_class.from_pretrained.return_value


def test_hf_loading_delegates_without_importing_native_mamba(monkeypatch):
    monkeypatch.setitem(sys.modules, "mamba_ssm", None)
    model = object.__new__(MambaLMWrapper)
    model.is_hf = True

    with (
        patch.object(HFLM, "_get_config", autospec=True) as config_loader,
        patch.object(HFLM, "_create_model", autospec=True) as model_loader,
    ):
        model._get_config("local-hf", revision="test-revision")
        model._create_model("local-hf", dtype="float32", trust_remote_code=True)

    config_loader.assert_called_once_with(model, "local-hf", revision="test-revision")
    model_loader.assert_called_once_with(
        model, "local-hf", dtype="float32", trust_remote_code=True
    )
