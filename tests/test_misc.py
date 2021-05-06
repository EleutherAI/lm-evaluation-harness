import pytest
import lm_eval.metrics as metrics


def test_bootstrapping():
    arr = list(range(100))
    expected = metrics.mean_stderr(arr)
    bootstrapped = metrics.bootstrap_stderr(metrics.mean, arr)

    assert bootstrapped == pytest.approx(expected)
