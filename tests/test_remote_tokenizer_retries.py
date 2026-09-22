import pytest
import requests

from lm_eval.utils import RemoteTokenizer


@pytest.mark.parametrize(("timeout_override", "expected_timeout"), [(None, 30), (7, 7)])
def test_retry_preserves_timeout(timeout_override, expected_timeout):
    class DummyResponse:
        def raise_for_status(self):
            pass

    response = DummyResponse()
    timeouts = []

    class FlakySession:
        def request(self, method, url, **kwargs):
            timeouts.append(kwargs["timeout"])
            if len(timeouts) == 1:
                raise requests.ConnectionError("transient failure")
            return response

    tokenizer = object.__new__(RemoteTokenizer)
    tokenizer.session = FlakySession()
    tokenizer.timeout = 30
    tokenizer.cert_config = True
    tokenizer.max_retries = 3

    kwargs = {} if timeout_override is None else {"timeout": timeout_override}
    assert (
        tokenizer._request_with_retries("GET", "http://example.invalid", **kwargs)
        is response
    )
    assert timeouts == [expected_timeout, expected_timeout]
