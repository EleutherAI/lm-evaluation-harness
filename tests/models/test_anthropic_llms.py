from lm_eval.models.anthropic_llms import AnthropicChat


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
