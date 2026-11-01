"""Offline tests for the Mamba adapter's model-loading and generation contract."""

from unittest.mock import patch

import pytest


pytest.importorskip("torch")
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
