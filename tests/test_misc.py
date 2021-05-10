import pytest
import lm_eval.metrics as metrics
import random


def test_bootstrapping():
    random.seed(42)
    arr = [random.random() for _ in range(100)]
    expected = metrics.mean_stderr(arr)
    bootstrapped = metrics.bootstrap_stderr(metrics.mean, arr, iters=100000)

    assert bootstrapped == pytest.approx(expected, abs=1e-4)


def test_bootstrapping_stella():
    arr = [0.1, 0.3, 0.2, 0.25, 0.3, 0.1, 0.22]
    expected = metrics.mean_stderr(arr)
    bootstrapped = metrics.bootstrap_stderr(metrics.mean, arr, iters=100000)

    assert bootstrapped == pytest.approx(expected, abs=1e-5)