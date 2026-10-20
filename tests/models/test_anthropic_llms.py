import pytest

from lm_eval.models.anthropic_llms import AnthropicChat


@pytest.mark.parametrize("system_message", [None, "  Follow the instructions.\n"])
def test_anthropic_chat_preserves_conversation_messages(system_message):
    api = AnthropicChat(model="claude-test", tokenizer_backend=None)
    messages = [
        {"role": "user", "content": "  First question?\n"},
        {"role": "assistant", "content": "First answer.\n"},
        {"role": "user", "content": "Second question?"},
    ]
    if system_message is not None:
        messages.insert(0, {"role": "system", "content": system_message})

    payload = api._create_payload(messages, gen_kwargs={})

    assert payload["messages"] == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "  First question?\n"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "First answer.\n"}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Second question?"}],
        },
    ]
    if system_message is None:
        assert "system" not in payload
    else:
        assert payload["system"] == system_message


@pytest.mark.parametrize("attribute", ["api_key", "header"])
def test_anthropic_chat_requires_api_key(monkeypatch, attribute):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    api = AnthropicChat(model="claude-test", tokenizer_backend=None)

    with pytest.raises(ValueError, match="Please set the ANTHROPIC_API_KEY"):
        getattr(api, attribute)


def test_anthropic_chat_builds_authentication_headers(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    api = AnthropicChat(model="claude-test", tokenizer_backend=None)

    assert api.header == {
        "x-api-key": "test-anthropic-key",
        "anthropic-version": "2023-06-01",
    }


def test_anthropic_chat_uses_default_stop_when_until_is_missing():
    api = AnthropicChat(model="claude-test", tokenizer_backend=None)

    payload = api._create_payload(
        [{"role": "user", "content": "hello"}],
        gen_kwargs={},
    )

    assert payload["stop_sequences"] == ["\n\nHuman:"]


def test_anthropic_chat_uses_default_stop_when_until_is_whitespace_only():
    api = AnthropicChat(model="claude-test", tokenizer_backend=None)

    payload = api._create_payload(
        [{"role": "user", "content": "hello"}],
        gen_kwargs={"until": ["\n\n", "  "]},
    )

    assert payload["stop_sequences"] == ["\n\nHuman:"]
