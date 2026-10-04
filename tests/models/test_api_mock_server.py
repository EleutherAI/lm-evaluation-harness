"""Run the OpenAI-compatible API models against a local mock server.

``test_api.py`` checks the payloads ``TemplateAPI`` builds. These tests check what
it does with the responses. A stdlib HTTP server on an ephemeral localhost port
answers with canned OpenAI-shaped responses, and the request paths run unpatched:
``requests`` when ``num_concurrent=1``, batched ``aiohttp`` calls when it is
higher, and the tenacity retries around both.
"""

import itertools
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import aiohttp
import pytest
import requests
import tenacity
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from lm_eval.api.instance import Instance
from lm_eval.models import api_models
from lm_eval.models.openai_completions import LocalChatCompletion, LocalCompletionsAPI


# The canned model. Every word has a fixed logprob; the values are exact binary
# fractions, so sums compare exactly. "dog" is never the model's top choice.
LOGPROBS = {
    "the": -0.5,
    "cat": -1.0,
    "sat": -0.25,
    "on": -0.125,
    "mat": -2.0,
    "a": -0.75,
    "dog": -4.0,
    "barked": -1.5,
}
NOT_GREEDY = {"dog"}
UNK, EOS = "<unk>", "<eos>"
VOCAB = [UNK, EOS, *LOGPROBS]
# The max_tokens=1 token generated after an echoed prompt. It must never be scored.
NEXT_TOKEN, NEXT_TOKEN_LOGPROB = EOS, -8.0
THINK_END = "</think>"

COMPLETIONS = {
    "Q: Capital of France?\nA:": " Paris; on the Seine.\nQ: Capital of Spain?",
    "Q: Capital of Spain?\nA:": " Madrid; on the Manzanares.\nQ: Capital of Italy?",
    "Translate to French: cat =>": " chat; dog => chien\n",
}
CHAT_REPLIES = {
    "Capital of France?": "Paris\nIt is also the largest city in France.",
    "Capital of Spain? Think first.": f"<think>Since 1561.{THINK_END} Madrid",
    "Say something unsafe.": None,  # e.g. blocked by a content filter
}

# ((context, continuation), (sum of continuation logprobs, is_greedy))
LOGLIKELIHOOD_CASES = [
    (("the", " cat"), (-1.0, True)),
    (("a dog", " barked on the mat"), (-4.125, True)),  # "dog" is context only
    (("a", " dog barked"), (-5.5, False)),
    (("the cat sat", " on"), (-0.125, True)),
]
# ((context, gen_kwargs), generated text). Each completion runs past the other
# group's stop sequence, so mixing up the `until` of the two groups changes the text.
GENERATE_CASES = [
    (("Q: Capital of France?\nA:", {"until": ["\n"]}), " Paris; on the Seine."),
    (("Translate to French: cat =>", {"until": [";"]}), " chat"),
    (("Q: Capital of Spain?\nA:", {"until": ["\n"]}), " Madrid; on the Manzanares."),
]
CHAT_CASES = [
    ("Capital of France?", "Paris"),
    ("Capital of Spain? Think first.", "Madrid"),
    ("Say something unsafe.", api_models.LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER),
]


def _apply_stop(text, stop):
    """Cut ``text`` before the first stop sequence, as an OpenAI server does."""
    if text is None:
        return None
    found = [i for i in (text.find(s) for s in stop or ()) if i >= 0]
    return text[: min(found, default=len(text))]


def _echo_choice(index, token_ids):
    """Answer an ``echo=True, logprobs=1, max_tokens=1`` loglikelihood request."""
    words = [VOCAB[t] for t in token_ids]
    # Nothing precedes the first prompt token, so it has no logprob.
    token_logprobs = [None] + [LOGPROBS[w] for w in words[1:]]
    # The top-1 entry is the token itself, unless a likelier one ("cat") beat it.
    top_logprobs = [None] + [
        {w: LOGPROBS[w], "cat": LOGPROBS[w] + 1.0}
        if w in NOT_GREEDY
        else {w: LOGPROBS[w]}
        for w in words[1:]
    ]
    return {
        "index": index,
        "text": " ".join([*words, NEXT_TOKEN]),
        "logprobs": {
            "tokens": [*words, NEXT_TOKEN],
            "token_logprobs": [*token_logprobs, NEXT_TOKEN_LOGPROB],
            "top_logprobs": [*top_logprobs, {NEXT_TOKEN: NEXT_TOKEN_LOGPROB}],
        },
        "finish_reason": "length",
    }


def _answer(path, payload):
    stop = payload.get("stop")
    if path == "/v1/chat/completions":
        content = CHAT_REPLIES[payload["messages"][-1]["content"]]
        message = {"role": "assistant", "content": _apply_stop(content, stop)}
        choice = {"index": 0, "message": message, "finish_reason": "stop"}
        return {"object": "chat.completion", "choices": [choice]}

    assert path == "/v1/completions", path
    prompts = payload["prompt"]
    if isinstance(prompts, str) or isinstance(prompts[0], int):
        prompts = [prompts]  # a single prompt rather than a batch
    if payload.get("echo"):
        choices = [_echo_choice(i, p) for i, p in enumerate(prompts)]
    else:
        choices = [
            {"index": i, "text": _apply_stop(COMPLETIONS[p], stop)}
            for i, p in enumerate(prompts)
        ]
    # Send the choices reversed: the parsers must place them by "index".
    return {"object": "text_completion", "choices": choices[::-1]}


class MockOpenAIServer:
    """Serves ``/v1/completions`` and ``/v1/chat/completions`` from the canned model.

    ``received`` and ``answered`` record request payloads in arrival and answer order.
    """

    def __init__(self):
        self.received = []
        self.answered = []
        self.max_in_flight = 0
        self._in_flight = 0
        self._finished = 0
        self._errors = iter(())
        self._hold, self._order_key = 0, None
        self._cond = threading.Condition()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.httpd.mock = self
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"

    def fail_requests(self, status, times=None):
        """Answer the next ``times`` requests (or all of them) with HTTP ``status``."""
        if times is None:
            self._errors = itertools.repeat(status)
        else:
            self._errors = itertools.repeat(status, times)

    def answer_in_order(self, n, key):
        """Hold the first ``n`` requests until all are in flight, then answer them sorted by ``key``."""
        self._hold, self._order_key = n, key

    def begin(self, payload):
        with self._cond:
            self.received.append(payload)
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
            if len(self.received) <= self._hold:
                self._cond.notify_all()
                # Time out rather than hang; the tests then fail on max_in_flight.
                self._cond.wait_for(lambda: len(self.received) >= self._hold, 3)
                order = sorted(self.received[: self._hold], key=self._order_key)
                turn = next(i for i, p in enumerate(order) if p is payload)
                self._cond.wait_for(lambda: self._finished >= turn, 3)
            self.answered.append(payload)
            return next(self._errors, 200)

    def end(self):
        with self._cond:
            self._in_flight -= 1
            self._finished += 1
            self._cond.notify_all()


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        mock = self.server.mock
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        status = mock.begin(payload)
        try:
            if status == 200:
                body = _answer(self.path, payload)
            else:
                body = {"error": {"message": "scripted failure", "code": status}}
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        finally:
            mock.end()

    def log_message(self, format, *args):
        pass


@pytest.fixture
def server():
    mock = MockOpenAIServer()
    # A short poll interval keeps shutdown() from waiting the default 0.5s.
    thread = threading.Thread(
        target=mock.httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    yield mock
    mock.httpd.shutdown()
    mock.httpd.server_close()
    thread.join()


@pytest.fixture(scope="module")
def word_tokenizer(tmp_path_factory):
    """A whitespace word-level tokenizer over ``VOCAB``, built locally (no Hub download)."""
    tokenizer = Tokenizer(WordLevel({w: i for i, w in enumerate(VOCAB)}, unk_token=UNK))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    path = tmp_path_factory.mktemp("word_tokenizer")
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token=UNK, eos_token=EOS
    ).save_pretrained(path)
    return str(path)


@pytest.fixture
def no_backoff(monkeypatch):
    """Skip the backoff between retries (at least 1s each); tenacity still retries."""
    monkeypatch.setattr(
        api_models, "wait_exponential", lambda **_: tenacity.wait_none()
    )


def completions_model(server, **kwargs):
    return LocalCompletionsAPI(
        base_url=f"{server.url}/completions", model="mock-model", **kwargs
    )


def make_instances(request_type, arguments):
    return [
        Instance(request_type=request_type, doc={}, arguments=args, idx=0)
        for args in arguments
    ]


def call_with_deadline(fn, *args, deadline=20):
    """Call ``fn`` in a daemon thread so a hang fails the test instead of stalling the suite.

    Broken concurrency or retry handling tends to hang (a leaked semaphore slot,
    a retry loop that never stops) rather than raise, so every model call goes
    through here.
    """
    outcome = {}

    def target():
        try:
            outcome["value"] = fn(*args)
        except BaseException as e:  # noqa: BLE001 - re-raised in the test thread
            outcome["error"] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(deadline)
    assert not thread.is_alive(), f"no result or error within {deadline}s"
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


@pytest.mark.parametrize(
    "batch_size, num_concurrent",
    [(1, 1), (2, 1), (1, 4), (2, 2)],
    ids=["sync", "sync-batched", "async", "async-batched"],
)
def test_loglikelihood_parses_echoed_logprobs(
    server, word_tokenizer, batch_size, num_concurrent
):
    model = completions_model(
        server,
        tokenizer=word_tokenizer,
        tokenizer_backend="huggingface",
        batch_size=batch_size,
        num_concurrent=num_concurrent,
    )

    results = call_with_deadline(
        model.loglikelihood,
        make_instances("loglikelihood", [args for args, _ in LOGLIKELIHOOD_CASES]),
    )

    assert results == [expected for _, expected in LOGLIKELIHOOD_CASES]
    assert len(server.received) == len(LOGLIKELIHOOD_CASES) // batch_size


def test_concurrent_results_keep_request_order(server, word_tokenizer):
    # The model sends the longest prompt first, as it sorts requests by length.
    # Answering the shortest first makes the responses complete in reverse.
    server.answer_in_order(4, key=lambda payload: len(payload["prompt"][0]))
    model = completions_model(
        server,
        tokenizer=word_tokenizer,
        tokenizer_backend="huggingface",
        num_concurrent=4,
    )

    results = call_with_deadline(
        model.loglikelihood,
        make_instances("loglikelihood", [args for args, _ in LOGLIKELIHOOD_CASES]),
    )

    assert server.max_in_flight == 4
    assert [len(p["prompt"][0]) for p in server.answered] == [2, 3, 4, 6]
    assert results == [expected for _, expected in LOGLIKELIHOOD_CASES]


@pytest.mark.parametrize(
    "batch_size, num_concurrent",
    [(1, 1), (2, 1), (1, 3)],
    ids=["sync", "sync-batched", "async"],
)
def test_generate_until_stops_at_each_requests_until(
    server, batch_size, num_concurrent
):
    model = completions_model(
        server,
        tokenizer_backend=None,
        batch_size=batch_size,
        num_concurrent=num_concurrent,
    )

    results = call_with_deadline(
        model.generate_until,
        make_instances("generate_until", [args for args, _ in GENERATE_CASES]),
    )

    assert results == [expected for _, expected in GENERATE_CASES]


@pytest.mark.parametrize("num_concurrent", [1, 2], ids=["sync", "async"])
def test_chat_generate_until_parses_message_content(server, num_concurrent):
    model = LocalChatCompletion(
        base_url=f"{server.url}/chat/completions",
        model="mock-model",
        think_end_token=THINK_END,
        num_concurrent=num_concurrent,
    )
    arguments = [
        (
            model.apply_chat_template([{"role": "user", "content": prompt}]),
            {"until": ["\n"]},
        )
        for prompt, _ in CHAT_CASES
    ]

    results = call_with_deadline(
        model.generate_until, make_instances("generate_until", arguments)
    )

    assert results == [expected for _, expected in CHAT_CASES]


@pytest.mark.parametrize("num_concurrent", [1, 2], ids=["sync", "async"])
def test_server_error_is_retried(server, no_backoff, num_concurrent):
    server.fail_requests(503, times=1)
    model = completions_model(
        server, tokenizer_backend=None, num_concurrent=num_concurrent, max_retries=3
    )
    args, expected = GENERATE_CASES[0]

    results = call_with_deadline(
        model.generate_until, make_instances("generate_until", [args])
    )

    assert results == [expected]
    assert len(server.received) == 2
    assert server.received[0] == server.received[1]


@pytest.mark.parametrize(
    "num_concurrent, error",
    [(1, requests.HTTPError), (2, aiohttp.ClientResponseError)],
    ids=["sync", "async"],
)
def test_persistent_server_error_raises_after_max_retries(
    server, no_backoff, num_concurrent, error
):
    server.fail_requests(500)
    model = completions_model(
        server, tokenizer_backend=None, num_concurrent=num_concurrent, max_retries=3
    )
    args, _ = GENERATE_CASES[0]

    with pytest.raises(error, match="500"):
        call_with_deadline(
            model.generate_until, make_instances("generate_until", [args])
        )

    assert len(server.received) == 3
