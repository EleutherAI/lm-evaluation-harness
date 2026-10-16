"""Test SGLang template metadata and cache identity without starting an engine."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lm_eval.api.model import TemplateLM
from lm_eval.api.task import Task
from lm_eval.config.task import TaskConfig
from lm_eval.loggers.evaluation_tracker import GeneralConfigTracker
from lm_eval.models.sglang_causallms import SGLangLM
from lm_eval.utils import hash_string


@pytest.fixture
def model():
    """Only metadata methods are exercised; no SGLang engine is required."""
    model = SGLangLM.__new__(SGLangLM)
    TemplateLM.__init__(model)
    model.tokenizer = SimpleNamespace(
        name_or_path="org/tokenizer",
        chat_template="{{ messages }}",
        default_chat_template=None,
    )
    return model


@pytest.mark.parametrize(
    ("name", "expected"),
    [("org/tokenizer-a", "org__tokenizer-a"), ("local-tokenizer", "local-tokenizer")],
)
def test_tokenizer_name(model, name, expected):
    """Cache identity must include the configured tokenizer name."""
    model.tokenizer.name_or_path = name
    assert model.tokenizer_name == expected


def test_chat_template_metadata(model):
    """The experiment tracker must receive the template and its fingerprint."""
    tracker = GeneralConfigTracker()
    tracker.log_experiment_args(
        model_source="sglang",
        model_args="pretrained=unused",
        system_instruction=None,
        chat_template=model.chat_template(True),
        fewshot_as_multiturn=False,
    )
    assert tracker.chat_template == model.tokenizer.chat_template
    assert tracker.chat_template_sha == hash_string(model.tokenizer.chat_template)


@pytest.mark.parametrize("setting", [False, None])
def test_disabled_template(model, setting):
    """Disabling the template follows the shared TemplateLM contract."""
    assert model.chat_template(setting) is None


@pytest.mark.parametrize(("setting", "expected"), [(True, "default"), ("tool", "tool")])
def test_named_chat_template(model, setting, expected):
    """Select either the default or an explicitly named tokenizer template."""
    model.tokenizer.chat_template = {"default": "default", "tool": "tool"}
    assert model.chat_template(setting) == expected


def test_unknown_chat_template(model):
    """Unknown names must raise rather than silently returning no metadata."""
    model.tokenizer.chat_template = {"default": "default"}
    with pytest.raises(ValueError, match="not available"):
        model.chat_template("missing")


def test_fallback_chat_template(model):
    """Tokenizers with a legacy default use the shared fallback behavior."""
    model.tokenizer.chat_template = None
    model.tokenizer.default_chat_template = "fallback"
    assert model.chat_template(True) == "fallback"


def test_request_cache_distinguishes_tokenizers(model, monkeypatch):
    """Two SGLang tokenizers must not load the same cached task requests."""
    load = Mock(return_value=[[object()]])
    monkeypatch.setattr("lm_eval.api.task.load_from_cache", load)
    config = TaskConfig(task="example", num_fewshot=0)
    task = SimpleNamespace(_config=config, config=config)
    for name in ("org/tokenizer-a", "org/tokenizer-b"):
        model.tokenizer.name_or_path = name
        Task.build_all_requests(
            task,
            cache_requests=True,
            apply_chat_template=True,
            tokenizer_name=model.tokenizer_name,
        )
    keys = [call.kwargs["file_name"] for call in load.call_args_list]
    assert len(keys) == 2
    assert keys[0] != keys[1]
    assert keys[0].endswith("-tokenizerorg__tokenizer-a")
    assert keys[1].endswith("-tokenizerorg__tokenizer-b")
