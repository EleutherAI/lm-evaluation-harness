"""CPU tests for vLLM thinking options passed through chat template arguments."""

from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import jinja2
import pytest
import transformers

from lm_eval.api.model import TemplateLM


THINK_END = "</think>"


@pytest.fixture
def vllm_backend(monkeypatch):
    """Stub the optional engine while running the real backend constructor."""
    import importlib.metadata
    import sys

    tokenizer = SimpleNamespace(
        name_or_path="test/tokenizer",
        pad_token_id=0,
        apply_chat_template=Mock(return_value="rendered"),
    )
    engine = SimpleNamespace(
        llm_engine=SimpleNamespace(model_config=SimpleNamespace(max_model_len=4096))
    )
    modules = {
        "vllm": {
            "LLM": Mock(return_value=engine),
            "SamplingParams": Mock(),
            "TokensPrompt": Mock(),
        },
        "vllm.lora": {},
        "vllm.lora.request": {"LoRARequest": Mock()},
        "vllm.renderers": {},
        "vllm.renderers.hf": {"resolve_chat_template": Mock(return_value=None)},
        "vllm.tokenizers": {"get_tokenizer": Mock(return_value=tokenizer)},
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__spec__ = ModuleSpec(name, loader=None)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    original_version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "0.18.0" if name == "vllm" else original_version(name),
    )
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        Mock(return_value=SimpleNamespace(model_type="test")),
    )
    spec = importlib.util.spec_from_file_location(
        "vllm_thinking_backend",
        Path(__file__).parents[2] / "lm_eval/models/vllm_causallms.py",
    )
    module = importlib.util.module_from_spec(spec)
    with patch("lm_eval.api.registry.register_model", return_value=lambda cls: cls):
        spec.loader.exec_module(module)
    return module.VLLM


@pytest.mark.parametrize("model_value", [None, False, True])
@pytest.mark.parametrize("template_value", [None, False, True])
def test_chat_template_thinking_precedence(vllm_backend, model_value, template_value):
    """An explicit template argument takes precedence over the model option."""
    template_args = {"custom_option": "kept"}
    if template_value is not None:
        template_args["enable_thinking"] = template_value
    expected = model_value if template_value is None else template_value
    model = vllm_backend(
        pretrained="unused",
        enable_thinking=model_value,
        chat_template_args=template_args,
        think_end_token=THINK_END,
    )
    history = [{"role": "user", "content": "hello"}]
    assert model.enable_thinking is expected
    assert model.apply_chat_template(history) == "rendered"
    model.tokenizer.apply_chat_template.assert_called_once_with(
        history,
        tokenize=False,
        add_generation_prompt=True,
        continue_final_message=False,
        chat_template=None,
        enable_thinking=expected,
        custom_option="kept",
    )


@pytest.mark.parametrize(
    ("model_value", "template_value", "raises"),
    [(None, True, True), (False, True, True), (True, False, False), (True, None, True)],
)
def test_thinking_validation_uses_resolved_value(
    vllm_backend, model_value, template_value, raises
):
    """The resolved option controls whether a thinking delimiter is required."""
    kwargs = {"pretrained": "unused", "enable_thinking": model_value}
    if template_value is not None:
        kwargs["chat_template_args"] = {"enable_thinking": template_value}
    if raises:
        with pytest.raises(ValueError, match="think_end_token"):
            vllm_backend(**kwargs)
    else:
        assert vllm_backend(**kwargs).enable_thinking is False


def test_template_thinking_rejects_loglikelihood(vllm_backend, monkeypatch):
    """Thinking enabled by the template must also enforce the scoring guard."""
    model = vllm_backend(
        pretrained="unused",
        chat_template_args={"enable_thinking": True},
        think_end_token=THINK_END,
    )
    monkeypatch.setattr(TemplateLM, "loglikelihood", Mock(return_value=[]))
    with pytest.raises(ValueError, match="not compatible with loglikelihood"):
        model.loglikelihood([])


def test_template_retry_preserves_thinking(vllm_backend):
    """Retrying without a system message must preserve the resolved option."""
    model = vllm_backend(
        pretrained="unused", chat_template_args={"enable_thinking": False}
    )
    model.tokenizer.apply_chat_template.side_effect = [
        jinja2.exceptions.TemplateError("system role unsupported"),
        "rendered",
    ]
    history = [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "hello"},
    ]
    assert model.apply_chat_template(history) == "rendered"
    calls = model.tokenizer.apply_chat_template.call_args_list
    assert len(calls) == 2
    assert calls[1].args == ([history[1]],)
    assert all(call.kwargs["enable_thinking"] is False for call in calls)
