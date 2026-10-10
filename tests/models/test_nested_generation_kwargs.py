"""Exercise nested generation-parameter grouping through the actual HTTP API path."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from lm_eval.api.instance import Instance
from lm_eval.models.openai_completions import LocalCompletionsAPI
from lm_eval.models.utils import Collator


@pytest.mark.parametrize("batch_size", [0, 2])
@pytest.mark.parametrize(
    "first, second",
    [
        ({"logit_bias": {"42": 10}}, {"logit_bias": {"42": -10}}),
        (
            {"response_format": {"type": "json_object"}},
            {"response_format": {"type": "text"}},
        ),
        ({"extra_body": [{"value": 1}]}, {"extra_body": [{"value": 2}]}),
        ({"until": "ab"}, {"until": ["a", "b"]}),
        ({"extra_body": {"key": "value"}}, {"extra_body": [["key", "value"]]}),
    ],
)
def test_generation_kwargs_preserve_nested_values(batch_size, first, second):
    samples = [("short", first), ("longer prompt", second), ("medium", first)]
    collator = Collator(samples, lambda x: -len(x[0]), group_by="gen_kwargs")
    batches = list(collator.get_batched(n=batch_size))
    assert all(all(item[1] == batch[0][1] for item in batch) for batch in batches)
    assert (
        collator.get_original([item for batch in batches for item in batch]) == samples
    )


def test_generation_kwargs_ignore_nested_mapping_order():
    first = {"logit_bias": {"42": 10, "17": -2}, "temperature": 0}
    second = {"temperature": 0, "logit_bias": {"17": -2, "42": 10}}
    samples = [("first", first), ("second", second)]
    collator = Collator(samples, group_by="gen_kwargs")
    assert len(list(collator.get_batched(n=0))) == 1


def test_generation_kwargs_preserve_list_tuple_equivalence():
    samples = [("first", {"until": ["END"]}), ("second", {"until": ("END",)})]
    collator = Collator(samples, group_by="gen_kwargs")
    assert len(list(collator.get_batched(n=0))) == 1


@pytest.mark.parametrize("num_concurrent", [1, 2])
def test_batched_generation_preserves_logit_bias(num_concurrent):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(payload)
            bias = payload["logit_bias"]["42"]
            response = json.dumps(
                {
                    "choices": [
                        {"index": index, "text": f"{prompt}:{bias}"}
                        for index, prompt in reversed(
                            list(enumerate(payload["prompt"]))
                        )
                    ]
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *_args):
            pass

    samples = [("short", 10), ("longer prompt", -10), ("medium", 10)]
    requests = [
        Instance(
            request_type="generate_until",
            doc={},
            arguments=(prompt, {"logit_bias": {"42": bias}, "max_gen_toks": 1}),
            idx=index,
        )
        for index, (prompt, bias) in enumerate(samples)
    ]
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            model = LocalCompletionsAPI(
                model="test-model",
                base_url=f"http://127.0.0.1:{server.server_port}/v1/completions",
                tokenizer_backend=None,
                batch_size=2,
                num_concurrent=num_concurrent,
                timeout=5,
                max_retries=1,
            )
            results = model.generate_until(requests, disable_tqdm=True)
            assert results == [f"{prompt}:{bias}" for prompt, bias in samples]
            assert sorted(
                (prompt, payload["logit_bias"]["42"])
                for payload in received
                for prompt in payload["prompt"]
            ) == sorted(samples)
        finally:
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()
