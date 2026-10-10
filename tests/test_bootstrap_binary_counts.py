import itertools
import math
import warnings
from collections import defaultdict
from unittest import mock

import numpy as np
import pytest

from lm_eval.api.metrics import (
    _binary_count_bootstrap,
    _binary_scores_from_counts,
    _bootstrap_internal_no_mp,
    bootstrap_stderr,
    f1_score,
    matthews_corrcoef,
    median,
    sample_stddev,
)


# (gold, pred) for tn, fp, fn, tp, the order the count helpers use
CELLS = [(0, 0), (0, 1), (1, 0), (1, 1)]
METRICS = [f1_score, matthews_corrcoef]


def _rows(counts):
    return [pair for pair, k in zip(CELLS, counts, strict=True) for _ in range(k)]


def _count_tables(n):
    for tn in range(n + 1):
        for fp in range(n - tn + 1):
            for fn in range(n - tn - fp + 1):
                yield [tn, fp, fn, n - tn - fp - fn]


@pytest.mark.parametrize("metric", METRICS)
def test_scores_from_counts_match_the_sklearn_aggregation(metric):
    """Every confusion table of 1 to 8 rows, degenerate ones included."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sklearn warns when F1 is undefined
        for n in range(1, 9):
            for counts in _count_tables(n):
                expected = metric(_rows(counts))
                got = _binary_scores_from_counts(metric, np.array([counts]))[0]
                assert math.isclose(got, expected, abs_tol=1e-12), counts


@pytest.mark.parametrize("metric", METRICS)
def test_count_draws_have_the_row_resampling_distribution(metric):
    """All 4**4 row resamples of a skewed sample, one cell empty, give the
    same score distribution as the multinomial over count tables.
    """
    base = [(0, 0), (0, 0), (0, 1), (1, 1)]
    n = len(base)
    by_rows = defaultdict(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for picks in itertools.product(range(n), repeat=n):
            score = float(metric([base[i] for i in picks]))
            by_rows[round(score, 12)] += 1 / n**n
    p = [base.count(cell) / n for cell in CELLS]
    by_counts = defaultdict(float)
    for counts in _count_tables(n):
        weight = math.factorial(n) / math.prod(math.factorial(c) for c in counts)
        weight *= math.prod(pi**c for pi, c in zip(p, counts, strict=True))
        if weight:
            score = float(_binary_scores_from_counts(metric, np.array([counts]))[0])
            by_counts[round(score, 12)] += weight
    assert set(by_rows) == set(by_counts)
    for score, prob in by_rows.items():
        assert math.isclose(prob, by_counts[score], abs_tol=1e-12), score


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize(
    "iters", [1, 250, 500, 999, 1000, 1001, 1500, 2500, 100_000, 100_001]
)
def test_draws_exactly_the_requested_replicates(metric, iters):
    """Honor the budget, including remainders of either batching boundary."""
    count_path = _binary_count_bootstrap(metric, _rows([30, 5, 7, 20]), iters)
    assert len(count_path) == iters


@pytest.mark.parametrize(
    ("metric", "items"),
    [
        (f1_score, [(0, 1), (2, 1), (1, 1)]),  # a label outside 0/1
        (f1_score, [("1", "0"), ("0", "0")]),  # string labels
        (matthews_corrcoef, [(0, 1, 1), (1, 1, 0)]),  # not pairs
        (matthews_corrcoef, []),  # nothing to count
        (median, [(0, 1), (1, 1)]),  # any other aggregation
    ],
)
def test_other_inputs_keep_the_row_path(metric, items):
    assert _binary_count_bootstrap(metric, items, 1000) is None


def test_bootstrap_stderr_starts_no_pool_and_repeats_exactly():
    items = _rows([400, 60, 90, 450])
    with mock.patch("multiprocessing.Pool", side_effect=AssertionError("pool")):
        first = bootstrap_stderr(matthews_corrcoef, items, 100_000)
        second = bootstrap_stderr(matthews_corrcoef, items, 100_000)
    assert first == second


@pytest.mark.parametrize("metric", METRICS)
def test_stderr_agrees_with_the_row_path(metric):
    """Different random streams, same distribution: agreement within
    Monte Carlo noise (1,000 row replicates carry about 2% of their own).
    """
    items = _rows([70, 12, 18, 100])
    with mock.patch("builtins.print"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        by_rows = sample_stddev(_bootstrap_internal_no_mp(metric, items, 1000))
    by_counts = sample_stddev(_binary_count_bootstrap(metric, items, 100_000).tolist())
    assert abs(by_counts - by_rows) / by_rows < 0.1
