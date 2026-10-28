"""Perspective API scoring for `realtoxicityprompts`, tested without network access.

Only urllib3's socket-level ``_make_request`` is replaced, so the session and adapter
setup in ``metric.py`` and urllib3's retry logic (status codes, ``Retry-After``,
backoff) all run for real.
"""

import io
import json
import logging
import math
import time
import traceback
from pathlib import Path

import pytest
import urllib3
from urllib3.connectionpool import HTTPConnectionPool
from urllib3.exceptions import NewConnectionError

from lm_eval.api.registry import get_aggregation
from lm_eval.tasks._yaml_loader import load_yaml


if int(urllib3.__version__.split(".")[0]) < 2:
    pytest.skip("the HTTP stub targets urllib3 2.x internals", allow_module_level=True)

TASK_YAML = (
    Path(__file__).parents[1]
    / "lm_eval/tasks/realtoxicityprompts/realtoxicityprompts.yaml"
)
API_KEY = "secret-test-key-123"


def ok(toxicity):
    body = {"attributeScores": {"TOXICITY": {"summaryScore": {"value": toxicity}}}}
    return 200, body, {}


def error(status, headers=None):
    return status, {"error": {"code": status}}, headers or {}


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv("PERSPECTIVE_API_KEY", API_KEY)
    monkeypatch.delenv("PERSPECTIVE_API_QPS", raising=False)
    # A proxy would make urllib3 open a real CONNECT tunnel before the stub is reached.
    for var in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv(var.lower(), raising=False)
    config = load_yaml(TASK_YAML)
    # Each test starts as if no request had been made yet.
    monkeypatch.setitem(metric_globals(config), "_last_request", -math.inf)
    return config


@pytest.fixture(autouse=True)
def sleeps(monkeypatch):
    """Record (instead of performing) throttling, backoff and Retry-After sleeps."""
    calls = []
    monkeypatch.setattr(time, "sleep", calls.append)
    return calls


@pytest.fixture
def logs(caplog, monkeypatch):
    # `setup_logging()` stops the "lm_eval" logger from propagating to caplog's handler.
    monkeypatch.setattr(logging.getLogger("lm_eval"), "propagate", True)
    caplog.set_level(logging.INFO)
    return caplog


class FakePerspectiveAPI:
    """Answers each HTTP attempt with the next scripted reply (the last one repeats)."""

    def __init__(self, monkeypatch, replies):
        self.replies = list(replies)
        self.requests = []
        self.request_times = []

        def make_request(pool, conn, method, url, **kwargs):
            self.requests.append((method, f"{pool.scheme}://{pool.host}{url}"))
            self.request_times.append(time.monotonic())
            reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
            if isinstance(reply, Exception):
                raise reply
            status, body, headers = reply
            return urllib3.HTTPResponse(
                body=io.BytesIO(json.dumps(body).encode()),
                headers={"Content-Type": "application/json", **headers},
                status=status,
                preload_content=False,
            )

        monkeypatch.setattr(HTTPConnectionPool, "_make_request", make_request)


def metric_globals(config):
    return config["process_results"].__globals__


def score(config, prediction):
    return config["process_results"]({}, [prediction])


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_transient_https_errors_are_retried(config, monkeypatch, status):
    api = FakePerspectiveAPI(monkeypatch, [error(status), ok(0.9)])

    result = score(config, "you are an idiot")

    assert result == {"score": 1, "perspective_api_toxicity_score": 0.9}
    assert len(api.requests) == 2
    assert all(m == "POST" and u.startswith("https://") for m, u in api.requests)


def test_retry_after_header_is_respected(config, sleeps, monkeypatch):
    FakePerspectiveAPI(monkeypatch, [error(429, {"Retry-After": "7"}), ok(0.1)])

    assert score(config, "hello")["perspective_api_toxicity_score"] == 0.1
    assert 7 in sleeps


def test_retry_after_cannot_shorten_the_qps_delay(config, sleeps, monkeypatch):
    FakePerspectiveAPI(monkeypatch, [error(429, {"Retry-After": "0"}), ok(0.1)])

    assert score(config, "hello")["perspective_api_toxicity_score"] == 0.1
    assert sleeps == [1]


def test_backoff_between_retries(config, sleeps, monkeypatch):
    FakePerspectiveAPI(monkeypatch, [error(503)])

    score(config, "hello")

    # Apply the QPS delay even when urllib3's first backoff would be zero.
    assert sleeps == [1, 2, 4, 8, 16]


def test_retries_and_next_document_observe_qps_after_slow_responses(
    config, sleeps, monkeypatch
):
    now = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(time, "sleep", sleep)
    api = FakePerspectiveAPI(monkeypatch, [error(503), error(503), ok(0.1), ok(0.2)])
    make_request = HTTPConnectionPool._make_request

    def slow_request(*args, **kwargs):
        response = make_request(*args, **kwargs)
        now[0] += 0.4
        return response

    monkeypatch.setattr(HTTPConnectionPool, "_make_request", slow_request)

    assert score(config, "first")["perspective_api_toxicity_score"] == 0.1
    assert score(config, "second")["perspective_api_toxicity_score"] == 0.2

    assert api.request_times == pytest.approx([0, 1.4, 3.8, 5.2])
    assert sleeps == pytest.approx([1, 2, 1])


def test_requests_are_throttled_to_the_configured_qps(config, sleeps, monkeypatch):
    monkeypatch.setenv("PERSPECTIVE_API_QPS", "4")
    api = FakePerspectiveAPI(monkeypatch, [ok(0.1)])

    assert score(config, "first")["perspective_api_toxicity_score"] == 0.1
    assert score(config, "second")["perspective_api_toxicity_score"] == 0.1

    assert len(api.requests) == 2
    # Nothing to wait for before the first request, a 1/4 s gap before the second.
    assert sleeps == [pytest.approx(0.25, abs=0.05)]


@pytest.mark.parametrize("since_last, expected", [(0.4, [0.6]), (5.0, [])])
def test_throttle_counts_time_since_the_last_request(
    config, sleeps, monkeypatch, since_last, expected
):
    api = FakePerspectiveAPI(monkeypatch, [ok(0.1)])
    monkeypatch.setitem(
        metric_globals(config), "_last_request", time.monotonic() - since_last
    )

    assert score(config, "hello")["perspective_api_toxicity_score"] == 0.1

    assert len(api.requests) == 1
    assert sleeps == [pytest.approx(s, abs=0.05) for s in expected]


@pytest.mark.parametrize(
    "reply, attempts",
    [
        pytest.param(error(429), 6, id="rate-limit-retries-exhausted"),
        pytest.param(error(503), 6, id="server-error-retries-exhausted"),
        pytest.param(NewConnectionError(None, "refused"), 1, id="connection-error"),
    ],
)
def test_failed_request_is_nan_and_key_is_not_logged(
    config, logs, monkeypatch, reply, attempts
):
    api = FakePerspectiveAPI(monkeypatch, [reply])

    result = score(config, "you are an idiot")

    # Not 0: an unscored prediction must not count as non-toxic.
    assert math.isnan(result["score"])
    assert math.isnan(result["perspective_api_toxicity_score"])
    assert len(api.requests) == attempts
    assert "No toxicity score could be retrieved" in logs.text
    assert API_KEY not in logs.text


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_rejected_request_raises_without_leaking_the_key(
    config, logs, monkeypatch, status
):
    # E.g. an invalid key or a wrong URL: every prediction would fail the same way.
    api = FakePerspectiveAPI(monkeypatch, [error(status)])

    with pytest.raises(metric_globals(config)["PerspectiveAPIError"]) as excinfo:
        score(config, "you are an idiot")

    assert len(api.requests) == 1
    message = str(excinfo.value)
    assert f"HTTP {status}" in message
    assert "?key=<PERSPECTIVE_API_KEY>" in message
    full_traceback = "".join(traceback.format_exception(excinfo.value))
    assert API_KEY not in full_traceback
    assert API_KEY not in logs.text


@pytest.mark.parametrize("prediction", ["", " \n "])
def test_empty_prediction_is_non_toxic_without_a_request(
    config, monkeypatch, prediction
):
    # The Perspective API rejects empty text with HTTP 400, which would stop the run.
    api = FakePerspectiveAPI(monkeypatch, [error(400)])

    assert score(config, prediction) == {
        "score": 0,
        "perspective_api_toxicity_score": 0.0,
    }
    assert api.requests == []


@pytest.mark.parametrize("value", [None, "0.4", True, math.nan, math.inf, -0.1, 1.1])
def test_invalid_successful_response_is_unscored(config, monkeypatch, value):
    FakePerspectiveAPI(monkeypatch, [ok(value)])

    result = score(config, "hello")

    assert math.isnan(result["score"])
    assert math.isnan(result["perspective_api_toxicity_score"])


def test_failed_requests_are_left_out_of_the_aggregate(config, monkeypatch):
    per_doc = []
    for reply in [ok(0.9), error(503), ok(0.1), ok(0.2)]:
        FakePerspectiveAPI(monkeypatch, [reply])
        per_doc.append(score(config, "some text"))

    aggregated = {
        m["metric"]: get_aggregation(m["aggregation"])(
            [r[m["metric"]] for r in per_doc]
        )
        for m in config["metric_list"]
    }

    assert aggregated["score"] == pytest.approx(1 / 3)
    assert aggregated["perspective_api_toxicity_score"] == pytest.approx(0.4)
