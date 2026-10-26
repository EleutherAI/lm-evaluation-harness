"""Generation stops should not change when API concurrency is enabled."""

import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from lm_eval.models import api_models
from lm_eval.models.api_models import JsonChatStr
from lm_eval.models.openai_completions import LocalChatCompletion, LocalCompletionsAPI


@pytest.mark.parametrize("model_cls", [LocalCompletionsAPI, LocalChatCompletion])
@pytest.mark.parametrize(
    "eos_string, tokenizer_eos, until, expected_stop",
    [
        ("<EOS>", None, None, ["<EOS>"]),
        ("<EOS>", None, ["STOP"], ["STOP", "<EOS>"]),
        ("<EOS>", None, ["STOP", "<EOS>"], ["STOP", "<EOS>"]),
        (None, None, ["STOP"], ["STOP"]),
        (None, None, None, []),
        (None, "<TOKENIZER_EOS>", None, ["<TOKENIZER_EOS>"]),
        ("<EOS>", "<TOKENIZER_EOS>", None, ["<EOS>"]),
    ],
    ids=[
        "configured-eos",
        "task-stop-and-eos",
        "eos-already-present",
        "task-stop-only",
        "no-stops",
        "tokenizer-eos",
        "configured-eos-overrides-tokenizer",
    ],
)
def test_sync_and_async_requests_use_the_same_stops(
    monkeypatch, model_cls, eos_string, tokenizer_eos, until, expected_stop
):
    model = model_cls(
        model="test-model",
        base_url="http://unused.invalid/v1/completions",
        tokenizer_backend=None,
        eos_string=eos_string,
    )
    if tokenizer_eos is not None:
        model.tokenizer_backend = "huggingface"
        model.tokenizer = SimpleNamespace(eos_token=tokenizer_eos)

    is_chat = model_cls is LocalChatCompletion
    messages = (
        [JsonChatStr(json.dumps([{"role": "user", "content": "Hello"}]))]
        if is_chat
        else ["Hello"]
    )
    choice = {"index": 0, "text": "answer"}
    if is_chat:
        choice = {"index": 0, "message": {"role": "assistant", "content": "answer"}}
    body = {"choices": [choice]}
    gen_kwargs = {"max_gen_toks": 8, "until": until}
    original_kwargs = copy.deepcopy(gen_kwargs)

    sync_response = MagicMock()
    sync_response.json.return_value = body
    post = MagicMock(return_value=sync_response)
    monkeypatch.setattr(api_models.requests, "post", post)
    model.model_call(messages, generate=True, gen_kwargs=gen_kwargs)
    sync_payload = post.call_args.kwargs["json"]

    async_response = MagicMock()
    async_response.json = AsyncMock(return_value=body)
    context = AsyncMock()
    context.__aenter__.return_value = async_response
    session = MagicMock()
    session.post.return_value = context

    async def run():
        return await model.amodel_call(
            session,
            asyncio.Semaphore(1),
            messages,
            generate=True,
            gen_kwargs=gen_kwargs,
        )

    assert asyncio.run(run()) == ["answer"]
    async_payload = session.post.call_args.kwargs["json"]
    assert sync_payload["stop"] == expected_stop
    assert async_payload["stop"] == expected_stop
    assert async_payload == sync_payload
    assert gen_kwargs == original_kwargs
