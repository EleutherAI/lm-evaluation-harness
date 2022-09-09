import pytest
import random

import lm_eval.api.metric as metrics
from lm_eval.api.utils import DEFAULT_SEED


def test_bootstrapping():
    random.seed(DEFAULT_SEED)
    arr = [random.random() for _ in range(1000)]
    expected = metrics.mean_stderr(arr)
    bootstrapped = metrics.bootstrap_stderr(metrics.mean, arr, iters=100000)

    assert bootstrapped == pytest.approx(expected, abs=1e-4)
