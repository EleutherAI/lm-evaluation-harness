"""The prompt budget and submitted completion limit must describe one request."""

import copy
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock

import pytest

from lm_eval.api.instance import Instance
from lm_eval.models.anthropic_llms import AnthropicChat
from lm_eval.models.openai_completions import (
    LocalChatCompletion,
    LocalCompletionsAPI,
    OpenAIChatCompletion,
)
from lm_eval.models.sglang_generate_API import SGLANGGENERATEAPI


@pytest.fixture
def completion_server():
    payloads = []

    class Handler(BaseHTTPRequestHandler):
        def respond(self, value):
            body = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            assert self.path == "/tokenizer_info"
            self.respond({"eos_token": None})

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/tokenize":
                self.respond({"tokens": list(map(ord, payload["prompt"]))})
            else:
                assert self.path == "/v1/completions"
                payloads.append(payload)
                self.respond(
                    {
                        "choices": [
                            {"index": index, "text": "".join(map(chr, prompt))}
                            for index, prompt in enumerate(payload["prompt"])
                        ]
                    }
                )

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1/completions", payloads
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


LIMIT_CASES = [
    ({"max_gen_toks": 2}, 2),
    ({"max_new_tokens": 2}, 2),
    ({"max_tokens": 2}, 2),
    ({"max_completion_tokens": 2}, 2),
    ({"max_gen_toks": 8}, 8),
    ({"max_new_tokens": 8}, 8),
    ({"max_tokens": 8}, 8),
    ({"max_completion_tokens": 8}, 8),
    ({"max_gen_toks": None, "max_tokens": 2}, 2),
    ({"max_new_tokens": 2, "max_tokens": 8}, 2),
    (
        {
            "max_gen_toks": 2,
            "max_new_tokens": 4,
            "max_tokens": 6,
            "max_completion_tokens": 8,
        },
        2,
    ),
    ({}, 6),
]


@pytest.mark.parametrize(("batch_size", "num_concurrent"), [(1, 1), (2, 1), (1, 2)])
@pytest.mark.parametrize(("limits", "expected_limit"), LIMIT_CASES)
def test_generation_budget_matches_http_payload(
    completion_server, batch_size, num_concurrent, limits, expected_limit
):
    base_url, payloads = completion_server
    model = LocalCompletionsAPI(
        base_url=base_url,
        model="test-model",
        tokenizer_backend="remote",
        max_length=13,  # TemplateAPI reserves one token for generation.
        max_gen_toks=6,
        batch_size=batch_size,
        num_concurrent=num_concurrent,
        max_retries=1,
    )
    # Keep a nonzero temperature even when do_sample is false: existing API
    # payload semantics are independent of the generation-length aliases.
    gen_kwargs = {**limits, "until": ["\n"], "temperature": 0.7, "do_sample": False}
    original = copy.deepcopy(gen_kwargs)
    prompts = ["abcdefghij", "ABCDEFGHIJKL"]
    requests = [
        Instance("generate_until", {}, (prompt, gen_kwargs), 0) for prompt in prompts
    ]
    try:
        outputs = model.generate_until(requests)
    finally:
        model.tokenizer.session.close()

    expected_prompts = [
        prompt[-(model.max_length - expected_limit) :] for prompt in prompts
    ]
    assert outputs == expected_prompts
    assert gen_kwargs == original
    assert len(payloads) == (1 if batch_size == 2 else 2)
    for payload in payloads:
        assert payload["max_tokens"] == expected_limit
        assert payload["temperature"] == 0.7
        assert (
            not {"max_gen_toks", "max_new_tokens", "max_completion_tokens"}
            & payload.keys()
        )
        assert all(
            len(prompt) + payload["max_tokens"] <= model.max_length
            for prompt in payload["prompt"]
        )


@pytest.mark.parametrize(
    ("model_cls", "limit_key"),
    [
        (LocalCompletionsAPI, "max_tokens"),
        (LocalChatCompletion, "max_tokens"),
        (OpenAIChatCompletion, "max_completion_tokens"),
        (AnthropicChat, "max_tokens"),
        (SGLANGGENERATEAPI, "max_new_tokens"),
    ],
)
@pytest.mark.parametrize(("limits", "expected_limit"), LIMIT_CASES)
def test_completion_payload_resolves_length_aliases(
    model_cls, limit_key, limits, expected_limit, monkeypatch
):
    model = model_cls(
        base_url="http://test.invalid/v1/completions",
        model="gpt-4o",
        tokenizer_backend=None,
        max_gen_toks=6,
    )
    messages = (
        [[1, 2]]
        if model_cls in (LocalCompletionsAPI, SGLANGGENERATEAPI)
        else [{"role": "user", "content": "hello"}]
    )
    if model_cls in (AnthropicChat, SGLANGGENERATEAPI):
        post = Mock()
        monkeypatch.setattr("lm_eval.models.api_models.requests.post", post)
        model.header = {}
        call_messages = (
            [model.apply_chat_template(messages)]
            if model_cls is AnthropicChat
            else ["hello"]
        )
        model.model_call(call_messages, generate=True, gen_kwargs=copy.deepcopy(limits))
        payload = post.call_args.kwargs["json"]
    else:
        payload = model._create_payload(
            messages, generate=True, gen_kwargs=copy.deepcopy(limits)
        )

    payload = payload.get("sampling_params", payload)
    assert payload[limit_key] == expected_limit
    assert {
        "max_gen_toks",
        "max_new_tokens",
        "max_tokens",
        "max_completion_tokens",
    } & payload.keys() == {limit_key}
