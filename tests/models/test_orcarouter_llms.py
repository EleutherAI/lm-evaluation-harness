import json

import pytest

from lm_eval.models.api_models import JsonChatStr
from lm_eval.models.orcarouter_llms import OrcaRouterChatCompletion


OPENAI_CHAT_RESPONSE = {
    "id": "chatcmpl-orca-test",
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello there!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
}


def _chat_messages(content="Hi"):
    """Wrap chat messages as JsonChatStr — the format generate_until uses."""
    return (JsonChatStr(json.dumps([{"role": "user", "content": content}])),)


@pytest.fixture
def orca_model(monkeypatch):
    monkeypatch.setenv("ORCAROUTER_API_KEY", "sk-orca-test")
    return OrcaRouterChatCompletion(model="orcarouter/auto")


def test_init_default_base_url(orca_model):
    assert orca_model.model == "orcarouter/auto"
    assert orca_model.base_url == "https://api.orcarouter.ai/v1/chat/completions"
    assert orca_model.tokenizer is None
    assert orca_model.tokenized_requests is False
    assert orca_model._batch_size == 1


def test_init_custom_args():
    model = OrcaRouterChatCompletion(
        model="openai/gpt-5.5",
        base_url="https://api.orcarouter.ai/v1/chat/completions",
        max_gen_toks=512,
        num_concurrent=4,
        seed=42,
    )
    assert model.model == "openai/gpt-5.5"
    assert model._max_gen_toks == 512
    assert model._concurrent == 4
    assert model._seed == 42


def test_api_key_from_env(orca_model):
    assert orca_model.api_key == "sk-orca-test"


def test_api_key_requires_env(monkeypatch):
    monkeypatch.delenv("ORCAROUTER_API_KEY", raising=False)
    model = OrcaRouterChatCompletion(model="orcarouter/auto")
    with pytest.raises(ValueError, match="ORCAROUTER_API_KEY"):
        _ = model.api_key


def test_create_payload(orca_model):
    messages = [{"role": "user", "content": "Hello"}]
    gen_kwargs = {
        "max_tokens": 100,
        "temperature": 0.7,
        "until": ["The End"],
        "do_sample": True,
    }
    payload = orca_model._create_payload(
        messages, generate=True, gen_kwargs=gen_kwargs, seed=1234
    )

    assert payload == {
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "orcarouter/auto",
        "max_tokens": 100,
        "temperature": 0.7,
        "stop": ["The End"],
        "seed": 1234,
    }


def test_create_payload_defaults(orca_model):
    messages = [{"role": "user", "content": "Hello"}]
    payload = orca_model._create_payload(
        messages, generate=True, gen_kwargs={}, seed=1234
    )

    assert payload["model"] == "orcarouter/auto"
    assert payload["max_tokens"] == 256
    assert payload["temperature"] == 0
    assert payload["seed"] == 1234


def test_parse_generations(orca_model):
    result = orca_model.parse_generations(OPENAI_CHAT_RESPONSE)
    assert result == ["Hello there!"]


def test_parse_generations_multiple_choices(orca_model):
    multi_response = {
        "choices": [
            {"index": 0, "message": {"content": "First"}},
            {"index": 1, "message": {"content": "Second"}},
        ]
    }
    result = orca_model.parse_generations(multi_response)
    assert result == ["First", "Second"]


def test_loglikelihood_raises(orca_model):
    with pytest.raises(NotImplementedError, match="Loglikelihood"):
        orca_model.loglikelihood([])


def test_tok_encode(orca_model):
    assert orca_model.tok_encode("hello world") == "hello world"


def test_registry_resolution():
    """Model names resolve through the lazy registry."""
    from lm_eval.api.registry import get_model

    assert get_model("orcarouter-chat-completions") is OrcaRouterChatCompletion
    assert get_model("orcarouter") is OrcaRouterChatCompletion
