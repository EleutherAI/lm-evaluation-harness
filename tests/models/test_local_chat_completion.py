from unittest.mock import Mock

import pytest
import requests

from lm_eval.api.instance import Instance
from lm_eval.models.openai_completions import LocalChatCompletion


@pytest.mark.parametrize(
    "think_end_token, content, expected",
    [
        (
            None,
            "<think>reasoning</think> final answer",
            "<think>reasoning</think> final answer",
        ),
        ("</think>", "<think>reasoning</think>  final answer", "final answer"),
        ("</think>", "first</think> draft</think> final answer", "final answer"),
        (
            "</think>",
            "answer without a thinking marker",
            "answer without a thinking marker",
        ),
        ("</think>", None, None),
    ],
)
def test_parse_generations_strips_thinking(think_end_token, content, expected):
    model = LocalChatCompletion(
        base_url="http://test-url.com",
        model="test-model",
        think_end_token=think_end_token,
    )
    response = {
        "choices": [{"index": 0, "message": {"content": content}}],
    }

    assert model.parse_generations(response) == [expected]


@pytest.fixture
def chat_request():
    def make_request(model):
        messages = [{"role": "user", "content": "What is six times seven?"}]
        return Instance(
            request_type="generate_until",
            doc={},
            arguments=(
                model.apply_chat_template(messages),
                {
                    "max_gen_toks": 512,
                    "temperature": 0,
                    "until": ["Q:", "</s>", "<|im_end|>"],
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            ),
            idx=0,
        )

    return make_request


def test_chat_generation_sends_messages_and_bearer_token(monkeypatch, chat_request):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-bearer-token")
    response = requests.Response()
    response.status_code = 200
    response._content = b'{"choices":[{"index":0,"message":{"content":"42"}}]}'
    post = Mock(return_value=response)
    monkeypatch.setattr(requests, "post", post)
    model = LocalChatCompletion(
        model="test-model",
        base_url="https://example.invalid/v1/chat/completions",
        tokenizer_backend=None,
        tokenized_requests=False,
        batch_size=1,
        num_concurrent=1,
        max_retries=1,
        timeout=17,
        seed=1234,
    )

    assert model.generate_until([chat_request(model)]) == ["42"]
    assert model.tokenizer is None
    post.assert_called_once_with(
        "https://example.invalid/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "What is six times seven?"}],
            "model": "test-model",
            "max_tokens": 512,
            "temperature": 0,
            "stop": ["Q:", "</s>", "<|im_end|>"],
            "seed": 1234,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        headers={"Authorization": "Bearer fixture-bearer-token"},
        verify=True,
        timeout=17,
    )


@pytest.mark.parametrize("status", [401, 429, 503])
def test_chat_one_attempt_propagates_http_error(monkeypatch, chat_request, status):
    response = requests.Response()
    response.status_code = status
    response._content = b'{"error":"synthetic failure"}'
    post = Mock(return_value=response)
    monkeypatch.setattr(requests, "post", post)
    model = LocalChatCompletion(
        model="test-model",
        base_url="https://example.invalid/v1/chat/completions",
        max_retries=1,
    )

    with pytest.raises(requests.HTTPError):
        model.generate_until([chat_request(model)])
    post.assert_called_once()


def test_chat_one_attempt_propagates_timeout(monkeypatch, chat_request):
    post = Mock(side_effect=requests.Timeout("synthetic timeout"))
    monkeypatch.setattr(requests, "post", post)
    model = LocalChatCompletion(
        model="test-model",
        base_url="https://example.invalid/v1/chat/completions",
        max_retries=1,
    )

    with pytest.raises(requests.Timeout):
        model.generate_until([chat_request(model)])
    post.assert_called_once()


def test_chat_rejects_loglikelihood():
    model = LocalChatCompletion(model="test-model")

    with pytest.raises(NotImplementedError, match="Loglikelihood is not supported"):
        model.loglikelihood([])
