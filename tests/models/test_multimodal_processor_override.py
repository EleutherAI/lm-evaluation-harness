"""Exercise processor overrides through actual HF multimodal initialization."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import transformers

from lm_eval.models.hf_vlms import HFMultimodalLM
from lm_eval.models.huggingface import HFLM


@pytest.fixture(params=["vision", "audio"])
def backend(request):
    """Load audio without touching its unrelated lazy-registration mismatch."""
    if request.param == "vision":
        return HFMultimodalLM
    spec = importlib.util.spec_from_file_location(
        "processor_override_audio_backend",
        Path(__file__).parents[2] / "lm_eval/models/hf_audiolm.py",
    )
    module = importlib.util.module_from_spec(spec)
    with patch("lm_eval.api.registry.register_model", return_value=lambda cls: cls):
        spec.loader.exec_module(module)
    return module.HFAUDIOLMQWEN


@pytest.fixture
def processor():
    """Provide a ProcessorMixin-compatible object with a lightweight tokenizer."""
    processor = Mock(spec=transformers.ProcessorMixin)
    processor.tokenizer = SimpleNamespace(vocab_size=32, pad_token_id=0)
    return processor


@pytest.mark.parametrize("selection", ["default", "path", "object"])
def test_processor_selection_initializes_both_attributes(
    backend, processor, selection, monkeypatch
):
    """HFLM ignores _create_tokenizer's return value, so it must set state."""
    load_processor = Mock(return_value=processor)
    load_tokenizer = Mock(return_value=processor.tokenizer)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", load_processor)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(HFLM, "_get_backend", Mock())
    pretrained = SimpleNamespace(
        device=torch.device("cpu"),
        config=SimpleNamespace(model_type="test"),
        name_or_path="model/loaded",
    )
    tokenizer = {"default": None, "path": "processor/custom", "object": processor}[
        selection
    ]
    extra = {"image_string": "<image>"} if backend is HFMultimodalLM else {}
    model = backend(pretrained=pretrained, tokenizer=tokenizer, **extra)
    assert model.processor is processor
    assert model.tokenizer is processor.tokenizer
    assert model.vocab_size == 32
    load_tokenizer.assert_not_called()
    if selection == "object":
        load_processor.assert_not_called()
    else:
        load_processor.assert_called_once_with(
            "processor/custom" if selection == "path" else "model/loaded",
            revision="main",
            trust_remote_code=False,
        )


def test_default_processor_loading_preserves_options(backend, processor, monkeypatch):
    """Keep default model-name resolution and vision pixel options intact."""
    load = Mock(return_value=processor)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", load)
    model = backend.__new__(backend)
    model.pixels = {"min_pixels": 32}
    model._create_tokenizer(
        "model/source", tokenizer=None, revision="custom", trust_remote_code=True
    )
    expected = {"min_pixels": 32} if backend is HFMultimodalLM else {}
    load.assert_called_once_with(
        "model/source", revision="custom", trust_remote_code=True, **expected
    )
    assert model.processor is processor
    assert model.tokenizer is processor.tokenizer
