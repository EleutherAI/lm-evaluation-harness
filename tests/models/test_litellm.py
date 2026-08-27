import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest


pytest.importorskip("litellm")

from lm_eval.models.api_models import JsonChatStr
from lm_eval.models.litellm_llms import LiteLLMChatCompletion


OPENAI_CHAT_RESPONSE = {
    "id": "chatcmpl-test",
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
    return (
        JsonChatStr(json.dumps([{"role": "user", "content": content}])),
    )


@pytest.fixture
def litellm_model():
    return LiteLLMChatCompletion(model="gpt-4o-mini")


def test_init(litellm_model):
    assert litellm_model.model == "gpt-4o-mini"
    assert litellm_model.tokenizer is None
    assert litellm_model.tokenized_requests is False
    assert litellm_model._batch_size == 1
    assert litellm_model._litellm is not None


def test_init_custom_args():
    model = LiteLLMChatCompletion(
        model="anthropic/claude-3-sonnet",
        max_gen_toks=512,
        num_concurrent=4,
        seed=42,
    )
    assert model.model == "anthropic/claude-3-sonnet"
    assert model._max_gen_toks == 512
    assert model._concurrent == 4
    assert model._seed == 42


def test_create_payload(litellm_model):
    messages = [{"role": "user", "content": "Hello"}]
    gen_kwargs = {
        "max_tokens": 100,
        "temperature": 0.7,
        "until": ["The End"],
        "do_sample": True,
    }
    payload = litellm_model._create_payload(
        messages, generate=True, gen_kwargs=gen_kwargs, seed=1234
    )

    assert payload == {
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "temperature": 0.7,
        "stop": ["The End"],
        "seed": 1234,
    }


def test_create_payload_defaults(litellm_model):
    messages = [{"role": "user", "content": "Hello"}]
    payload = litellm_model._create_payload(
        messages, generate=True, gen_kwargs={}, seed=1234
    )

    assert payload["model"] == "gpt-4o-mini"
    assert payload["max_tokens"] == 256
    assert payload["temperature"] == 0
    assert payload["seed"] == 1234


def test_model_call(litellm_model):
    mock_response = MagicMock()
    mock_response.model_dump.return_value = OPENAI_CHAT_RESPONSE

    litellm_model._litellm = MagicMock()
    litellm_model._litellm.completion.return_value = mock_response

    result = litellm_model.model_call(
        _chat_messages("Hi"),
        generate=True,
        gen_kwargs={"max_tokens": 50, "temperature": 0},
    )

    litellm_model._litellm.completion.assert_called_once()
    call_kwargs = litellm_model._litellm.completion.call_args[1]
    assert call_kwargs["model"] == "gpt-4o-mini"
    assert call_kwargs["messages"] == [{"role": "user", "content": "Hi"}]
    assert result == OPENAI_CHAT_RESPONSE


def test_model_call_error_propagates(litellm_model):
    litellm_model._litellm = MagicMock()
    litellm_model._litellm.completion.side_effect = RuntimeError("Rate limited")

    with pytest.raises(RuntimeError, match="Rate limited"):
        litellm_model.model_call(_chat_messages(), generate=True, gen_kwargs={})


def test_amodel_call():
    model = LiteLLMChatCompletion(model="gpt-4o-mini")

    mock_response = MagicMock()
    mock_response.model_dump.return_value = OPENAI_CHAT_RESPONSE

    model._litellm = MagicMock()
    model._litellm.acompletion = AsyncMock(return_value=mock_response)

    sem = asyncio.Semaphore(1)

    async def run():
        return await model.amodel_call(
            session=None,
            sem=sem,
            messages=_chat_messages("Hi"),
            generate=True,
            gen_kwargs={"max_tokens": 50, "temperature": 0},
        )

    result = asyncio.run(run())

    model._litellm.acompletion.assert_called_once()
    assert result == ["Hello there!"]


def test_amodel_call_with_caching():
    model = LiteLLMChatCompletion(model="gpt-4o-mini")

    mock_response = MagicMock()
    mock_response.model_dump.return_value = OPENAI_CHAT_RESPONSE

    model._litellm = MagicMock()
    model._litellm.acompletion = AsyncMock(return_value=mock_response)
    model.cache_hook = MagicMock()

    sem = asyncio.Semaphore(1)
    cache_keys = [("Hi", {"max_tokens": 50})]

    async def run():
        return await model.amodel_call(
            session=None,
            sem=sem,
            messages=_chat_messages("Hi"),
            generate=True,
            cache_keys=cache_keys,
            gen_kwargs={"max_tokens": 50, "temperature": 0},
        )

    result = asyncio.run(run())

    model.cache_hook.add_partial.assert_called_once_with(
        "generate_until", ("Hi", {"max_tokens": 50}), "Hello there!"
    )
    assert result == ["Hello there!"]


def test_amodel_call_releases_semaphore_on_error():
    model = LiteLLMChatCompletion(model="gpt-4o-mini")

    model._litellm = MagicMock()
    model._litellm.acompletion = AsyncMock(side_effect=RuntimeError("API down"))

    sem = asyncio.Semaphore(1)

    async def run():
        with pytest.raises(RuntimeError, match="API down"):
            await model.amodel_call(
                session=None,
                sem=sem,
                messages=_chat_messages(),
                generate=True,
                gen_kwargs={},
            )
        assert not sem.locked()

    asyncio.run(run())


def test_loglikelihood_raises(litellm_model):
    with pytest.raises(NotImplementedError, match="Loglikelihood is not supported"):
        litellm_model.loglikelihood([])


def test_loglikelihood_rolling_raises(litellm_model):
    with pytest.raises(NotImplementedError, match="Rolling loglikelihood"):
        litellm_model.loglikelihood_rolling([])


def test_parse_generations(litellm_model):
    result = litellm_model.parse_generations(OPENAI_CHAT_RESPONSE)
    assert result == ["Hello there!"]


def test_parse_generations_multiple_choices(litellm_model):
    multi_response = {
        "choices": [
            {"index": 0, "message": {"content": "First"}},
            {"index": 1, "message": {"content": "Second"}},
        ]
    }
    result = litellm_model.parse_generations(multi_response)
    assert result == ["First", "Second"]


def test_tok_encode(litellm_model):
    assert litellm_model.tok_encode("hello world") == "hello world"
