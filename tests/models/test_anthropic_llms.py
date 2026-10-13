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


def test_anthropic_chat_sends_temperature_to_models_that_accept_it():
    api = AnthropicChat(model="claude-sonnet-4-6", tokenizer_backend=None)

    payload = api._create_payload(
        [{"role": "user", "content": "hello"}],
        gen_kwargs={"temperature": 0, "top_p": 0.9},
    )

    assert payload["temperature"] == 0
    assert payload["top_p"] == 0.9


def test_anthropic_chat_leaves_out_sampling_params_for_models_that_reject_them():
    for model in (
        "claude-opus-4-7",
        "claude-opus-5-5",
        "claude-sonnet-5-5",
        "claude-haiku-5-5",
        "claude-fable-5-1",
    ):
        api = AnthropicChat(model=model, tokenizer_backend=None)

        payload = api._create_payload(
            [{"role": "user", "content": "hello"}],
            gen_kwargs={"temperature": 0, "top_p": 0.9, "top_k": 5},
        )

        assert "temperature" not in payload, model
        assert "top_p" not in payload, model
        assert "top_k" not in payload, model


def test_anthropic_chat_returns_one_generation_per_response():
    api = AnthropicChat(model="claude-opus-5-5", tokenizer_backend=None)

    outputs = [
        {
            "content": [
                {"type": "thinking", "thinking": "", "signature": "s"},
                {"type": "text", "text": "Paris"},
            ]
        },
        {"content": [{"type": "text", "text": "Ber"}, {"type": "text", "text": "lin"}]},
        {"content": []},
    ]

    assert api.parse_generations(outputs) == ["Paris", "Berlin", ""]
