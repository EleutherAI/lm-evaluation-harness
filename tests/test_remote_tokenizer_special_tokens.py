import pytest

from lm_eval.models.openai_completions import LocalCompletionsAPI
from lm_eval.utils import RemoteTokenizer


def test_remote_tokenizer_forwards_add_special_tokens(monkeypatch):
    class DummyResponse:
        def json(self):
            return {"tokens": [1, 2, 3]}

    payloads = []
    tokenizer = object.__new__(RemoteTokenizer)
    tokenizer.base_url = "https://mock-server"

    def dummy_request(method, url, **kwargs):
        payloads.append(kwargs["json"])
        return DummyResponse()

    monkeypatch.setattr(tokenizer, "_request_with_retries", dummy_request)

    assert tokenizer("hello", add_special_tokens=True) == {"input_ids": [1, 2, 3]}
    assert payloads == [{"prompt": "hello", "add_special_tokens": True}]


@pytest.mark.parametrize(
    ("text", "expected_calls"),
    [
        ("hello", [("hello", True)]),
        (["hello", "world"], [("hello", True), ("world", True)]),
    ],
)
def test_api_remote_tokenizer_forwards_add_special_tokens(text, expected_calls):
    calls = []

    class DummyTokenizer:
        def encode(self, value, add_special_tokens=False):
            calls.append((value, add_special_tokens))
            return [1, 2, 3]

    api = object.__new__(LocalCompletionsAPI)
    api.tokenizer_backend = "remote"
    api.tokenizer = DummyTokenizer()

    api.tok_encode(text, add_special_tokens=True)

    assert calls == expected_calls
