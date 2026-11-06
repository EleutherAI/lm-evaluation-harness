import json

import pytest
import requests

from lm_eval.models.api_models import TemplateAPI
from lm_eval.models.openai_completions import LocalCompletionsAPI
from lm_eval.utils import check_remote_tokenizer_support


@pytest.fixture
def tokenizer_responses(monkeypatch):
    bodies = {
        "tokenizer_info": b'{"eos_token": "</s>"}',
        "tokenize": b'{"tokens": [1, 2, 3]}',
    }

    def request(self, method, url, **kwargs):
        response = requests.Response()
        response.status_code = 200
        response.url = url
        response._content = bodies[url.rsplit("/", 1)[-1]]
        return response

    monkeypatch.setattr(requests.Session, "request", request)
    return bodies


@pytest.mark.parametrize("endpoint", ["tokenizer_info", "tokenize"])
@pytest.mark.parametrize(
    "body",
    [
        b"<html>Not a tokenizer endpoint</html>",
        b"",
        b"null",
        b"[]",
        b'"text"',
        b"42",
        b"{}",
    ],
)
def test_remote_tokenizer_probe_rejects_invalid_responses(
    tokenizer_responses, endpoint, body
):
    tokenizer_responses[endpoint] = body
    assert (
        check_remote_tokenizer_support("https://example.test/v1/completions") is False
    )


@pytest.mark.parametrize("endpoint", ["tokenizer_info", "tokenize"])
@pytest.mark.parametrize("error_type", [json.JSONDecodeError, ValueError])
def test_remote_tokenizer_probe_handles_legacy_json_errors(
    monkeypatch, tokenizer_responses, endpoint, error_type
):
    original_json = requests.Response.json

    def legacy_json(self, **kwargs):
        if self.url.rsplit("/", 1)[-1] == endpoint:
            if error_type is json.JSONDecodeError:
                raise json.JSONDecodeError("Expecting value", self.text, 0)
            raise ValueError("Invalid JSON")
        return original_json(self, **kwargs)

    monkeypatch.setattr(requests.Response, "json", legacy_json)
    assert check_remote_tokenizer_support("https://example.test") is False


def test_remote_tokenizer_probe_accepts_valid_responses(tokenizer_responses):
    assert check_remote_tokenizer_support("https://example.test/v1/completions") is True


@pytest.mark.parametrize("tokens", [b"null", b"42", b'"text"', b"{}"])
def test_remote_tokenizer_probe_rejects_non_list_tokens(tokenizer_responses, tokens):
    tokenizer_responses["tokenize"] = b'{"tokens": ' + tokens + b"}"
    assert check_remote_tokenizer_support("https://example.test") is False


@pytest.mark.parametrize("endpoint", ["tokenizer_info", "tokenize"])
def test_auto_tokenizer_falls_back_on_invalid_response(
    monkeypatch, tokenizer_responses, endpoint
):
    tokenizer_responses[endpoint] = b"<html>Not a tokenizer endpoint</html>"
    initialized = {}

    def initialize(self, **kwargs):
        initialized.update(kwargs)

    monkeypatch.setattr(TemplateAPI, "__init__", initialize)
    LocalCompletionsAPI(base_url="https://example.test/v1/completions")
    assert initialized["tokenizer_backend"] == "huggingface"
