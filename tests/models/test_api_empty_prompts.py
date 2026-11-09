import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lm_eval.models.openai_completions import LocalChatCompletion, LocalCompletionsAPI


TEXT_BATCHES = [
    [""],
    ["hello"],
    ["", "hello"],
    ["hello", ""],
    ["", ""],
]


def make_api(batch_size=1):
    return LocalCompletionsAPI(
        base_url="http://test-url.com",
        tokenizer_backend=None,
        model="test-model",
        batch_size=batch_size,
        eos_string="<end>",
        header={"X-Test-Fixture": "empty-prompts"},
    )


@pytest.mark.parametrize("messages", TEXT_BATCHES)
def test_model_call_preserves_empty_text_prompts(monkeypatch, messages):
    api = make_api(batch_size=len(messages))
    post = Mock()
    post.return_value.json.return_value = {"choices": []}
    monkeypatch.setattr("lm_eval.models.api_models.requests.post", post)

    api.model_call(messages, gen_kwargs={})

    post.assert_called_once()
    expected = messages[0] if len(messages) == 1 else messages
    assert post.call_args.kwargs["json"]["prompt"] == expected


@pytest.mark.parametrize("messages", TEXT_BATCHES)
def test_async_model_call_preserves_empty_text_prompts(messages):
    api = make_api(batch_size=len(messages))
    payloads = []
    choices = [{"index": i, "text": f"answer-{i}"} for i in range(len(messages))]

    async def response_json():
        return {"choices": choices}

    @asynccontextmanager
    async def post(url, *, json, headers):
        payloads.append(json)
        yield SimpleNamespace(
            ok=True, raise_for_status=lambda: None, json=response_json
        )

    async def run():
        return await api.amodel_call(
            SimpleNamespace(post=post),
            asyncio.Semaphore(1),
            messages,
            gen_kwargs={},
        )

    assert asyncio.run(run()) == [choice["text"] for choice in choices]
    assert len(payloads) == 1
    expected = messages[0] if len(messages) == 1 else messages
    assert payloads[0]["prompt"] == expected


@pytest.mark.parametrize("tokens", [[[]], [[1, 2]], [[], [1, 2]], [[1, 2], []]])
def test_create_message_decodes_empty_token_lists(tokens):
    api = make_api(batch_size=len(tokens))
    api.tokenizer_backend = "huggingface"
    api.tokenizer = Mock()
    decoded = ["" if not row else "hello" for row in tokens]
    api.tokenizer.batch_decode.return_value = decoded

    result = api.create_message(tokens)

    api.tokenizer.batch_decode.assert_called_once_with(tokens)
    assert result == (decoded[0] if len(tokens) == 1 else decoded)


@pytest.mark.parametrize("tokens", [[[]], [[1, 2]], [[], [1, 2]]])
def test_create_message_preserves_tokenized_requests(tokens):
    api = make_api(batch_size=len(tokens))
    api.tokenized_requests = True

    assert api.create_message(tokens) is tokens


def test_create_message_preserves_empty_chat_content():
    api = LocalChatCompletion(model="test-model")
    chat = [{"role": "user", "content": ""}]

    assert api.create_message((api.apply_chat_template(chat),)) == chat
