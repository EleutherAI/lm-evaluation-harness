from unittest.mock import Mock

import pytest

from lm_eval.tasks.realtoxicityprompts.metric import (
    DEFAULT_PERSPECTIVE_API_URL,
    toxicity_perspective_api,
)


@pytest.mark.parametrize(
    ("configured_url", "expected_url"),
    [
        (None, f"{DEFAULT_PERSPECTIVE_API_URL}?key=test-key"),
        (
            "https://proxy.example/v1/analyze?client=lm-eval",
            "https://proxy.example/v1/analyze?client=lm-eval&key=test-key",
        ),
    ],
)
def test_toxicity_perspective_api_url(monkeypatch, configured_url, expected_url):
    monkeypatch.setenv("PERSPECTIVE_API_KEY", "test-key")
    if configured_url is None:
        monkeypatch.delenv("PERSPECTIVE_API_URL", raising=False)
    else:
        monkeypatch.setenv("PERSPECTIVE_API_URL", configured_url)

    response = Mock(
        ok=True,
        text='{"attributeScores":{"TOXICITY":{"summaryScore":{"value":0.1}}}}',
    )
    session = Mock()
    session.post.return_value = response
    monkeypatch.setattr(
        "lm_eval.tasks.realtoxicityprompts.metric.requests.Session",
        lambda: session,
    )

    result = toxicity_perspective_api({}, ["safe response"], total_retries=1)

    assert session.post.call_args.args[0] == expected_url
    assert result == {"score": 0, "perspective_api_toxicity_score": 0.1}
