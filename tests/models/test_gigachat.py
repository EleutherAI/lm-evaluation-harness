import asyncio
import copy
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from lm_eval.models.api_models import JsonChatStr
from lm_eval.models.gigachat import GigaChatLM


RESPONSE = {
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Answer\nSTOPignored"},
            "finish_reason": "stop",
        }
    ]
}


@pytest.fixture
def model(monkeypatch):
    client = MagicMock()
    sync_response = MagicMock()
    sync_response.model_dump.side_effect = lambda **kwargs: copy.deepcopy(RESPONSE)
    async_response = MagicMock()
    async_response.model_dump.side_effect = lambda **kwargs: copy.deepcopy(RESPONSE)
    client.chat.return_value = sync_response
    client.achat = AsyncMock(return_value=async_response)
    client_cls = MagicMock(return_value=client)
    monkeypatch.setitem(sys.modules, "gigachat", SimpleNamespace(GigaChat=client_cls))

    instance = GigaChatLM(model="GigaChat-Test")
    instance.cache_hook = MagicMock()
    return instance, client, client_cls


def _messages(content="Hello"):
    return (JsonChatStr(json.dumps([{"role": "user", "content": content}])),)


def test_init_delegates_configuration_to_sdk(monkeypatch):
    client_cls = MagicMock()
    monkeypatch.setitem(sys.modules, "gigachat", SimpleNamespace(GigaChat=client_cls))

    GigaChatLM(
        model="GigaChat-Pro",
        credentials="secret",
        scope="GIGACHAT_API_B2B",
        ca_bundle_file="/certs/ca.pem",
        client_max_retries=5,
    )

    client_cls.assert_called_once_with(
        model="GigaChat-Pro",
        credentials="secret",
        scope="GIGACHAT_API_B2B",
        ca_bundle_file="/certs/ca.pem",
        max_retries=5,
    )


def test_create_payload_maps_generation_arguments(model):
    instance, _, _ = model

    payload = instance._create_payload(
        [{"role": "user", "content": "Hello"}],
        gen_kwargs={
            "max_gen_toks": 42,
            "temperature": 0.4,
            "top_p": 0.8,
            "until": ["STOP"],
            "do_sample": True,
        },
    )

    assert payload == {
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "GigaChat-Test",
        "max_tokens": 42,
        "temperature": 0.4,
        "top_p": 0.8,
        "_lm_eval_stop": ["STOP"],
    }


def test_model_call_uses_sdk_and_applies_stop(model):
    instance, client, _ = model

    result = instance.model_call(
        _messages(), gen_kwargs={"max_tokens": 20, "until": ["STOP"]}
    )

    client.chat.assert_called_once_with(
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "GigaChat-Test",
            "max_tokens": 20,
        }
    )
    assert result["choices"][0]["message"]["content"] == "Answer\n"


def test_do_sample_false_requests_greedy_decoding(model):
    instance, _, _ = model

    payload = instance._create_payload(
        [{"role": "user", "content": "Hello"}],
        gen_kwargs={"do_sample": False},
    )

    assert payload["temperature"] == 1.0
    assert payload["top_p"] == 0.0
    assert payload["repetition_penalty"] == 1.0


def test_async_model_call_uses_sdk_and_caches(model):
    instance, client, _ = model
    sem = asyncio.Semaphore(1)
    cache_key = ("Hello", {"until": ["STOP"]})

    result = asyncio.run(
        instance.amodel_call(
            session=None,
            sem=sem,
            messages=_messages(),
            cache_keys=[cache_key],
            gen_kwargs={"until": ["STOP"]},
        )
    )

    client.achat.assert_awaited_once()
    instance.cache_hook.add_partial.assert_called_once_with(
        "generate_until", cache_key, "Answer\n"
    )
    assert result == ["Answer\n"]
    assert not sem.locked()


def test_loglikelihood_is_not_supported(model):
    instance, _, _ = model

    with pytest.raises(NotImplementedError, match="Loglikelihood"):
        instance.loglikelihood([])
    with pytest.raises(NotImplementedError, match="Rolling loglikelihood"):
        instance.loglikelihood_rolling([])
