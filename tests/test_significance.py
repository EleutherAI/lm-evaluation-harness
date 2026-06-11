"""Tests for the paired significance utilities in ``lm_eval.api.significance``.

Expected values for the exact McNemar cases are computed by hand from the
binomial mass function so the tests do not depend on ``scipy``.
"""

import math

import numpy as np
import pytest

from lm_eval.api.significance import (
    compare_paired,
    mcnemar_test,
    normal_ci,
    paired_bootstrap_test,
)


def test_normal_ci_matches_known_z():
    low, high = normal_ci(0.5, 0.1, confidence=0.95)
    # 95% two-sided z = 1.959964
    assert low == pytest.approx(0.5 - 1.959964 * 0.1, abs=1e-5)
    assert high == pytest.approx(0.5 + 1.959964 * 0.1, abs=1e-5)
    # 99% two-sided z = 2.575829
    low99, high99 = normal_ci(0.0, 1.0, confidence=0.99)
    assert high99 == pytest.approx(2.575829, abs=1e-5)
    assert low99 == pytest.approx(-2.575829, abs=1e-5)


def test_normal_ci_rejects_bad_confidence():
    with pytest.raises(ValueError):
        normal_ci(0.5, 0.1, confidence=1.5)


def test_mcnemar_exact_all_discordant_one_side():
    # b=10, c=0 -> p = 2 * 0.5**10 = 2 / 1024
    res = mcnemar_test(10, 0)
    assert res.method == "mcnemar_exact"
    assert res.statistic is None
    assert res.p_value == pytest.approx(2.0 / 1024.0)
    assert res.n_discordant == 10


def test_mcnemar_exact_hand_value():
    # b=8, c=2, n=10 -> p = 2 * (C(10,0)+C(10,1)+C(10,2)) / 2**10
    #               = 2 * (1 + 10 + 45) / 1024 = 112 / 1024
    res = mcnemar_test(8, 2)
    assert res.p_value == pytest.approx(112.0 / 1024.0)


def test_mcnemar_symmetric_is_not_significant():
    res = mcnemar_test(7, 7)
    assert res.p_value == pytest.approx(1.0)


def test_mcnemar_no_discordant_pairs():
    res = mcnemar_test(0, 0)
    assert res.p_value == 1.0
    assert res.statistic == 0.0


def test_mcnemar_chi2_with_continuity_correction():
    # b=30, c=10, n=40, cc -> chi2 = (|20| - 1)**2 / 40 = 361 / 40 = 9.025
    res = mcnemar_test(30, 10, exact=False, continuity_correction=True)
    assert res.method == "mcnemar_chi2_cc"
    assert res.statistic == pytest.approx(9.025)
    # P(chi2_1 > 9.025) = erfc(sqrt(9.025 / 2))
    assert res.p_value == pytest.approx(math.erfc(math.sqrt(9.025 / 2.0)))
    assert res.p_value < 0.05


def test_mcnemar_rejects_negative_counts():
    with pytest.raises(ValueError):
        mcnemar_test(-1, 3)


def test_paired_bootstrap_is_deterministic():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, size=200).astype(float)
    b = rng.integers(0, 2, size=200).astype(float)
    r1 = paired_bootstrap_test(a, b, iters=2000, seed=42)
    r2 = paired_bootstrap_test(a, b, iters=2000, seed=42)
    assert r1 == r2
    # CI must bracket the observed difference.
    assert r1.ci_low <= r1.diff <= r1.ci_high


def test_paired_bootstrap_detects_clear_difference():
    # A correct everywhere, B correct on the first half only.
    a = np.ones(100)
    b = np.concatenate([np.ones(50), np.zeros(50)])
    res = paired_bootstrap_test(a, b, iters=5000, seed=7)
    assert res.diff == pytest.approx(0.5)
    assert res.ci_low > 0.0  # zero difference excluded
    assert res.p_value < 0.05


def test_paired_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap_test([0, 1, 1], [0, 1])


def test_compare_paired_auto_selects_mcnemar_for_binary():
    a = np.concatenate([np.ones(40), np.zeros(60)])
    b = np.concatenate([np.zeros(40), np.ones(60)])
    res = compare_paired(a, b)
    assert res.method.startswith("mcnemar")
    assert res.mcnemar is not None
    assert res.bootstrap is not None  # CI always present
    assert res.n == 100


def test_compare_paired_auto_selects_bootstrap_for_continuous():
    rng = np.random.default_rng(1)
    a = rng.normal(0.7, 0.1, size=300)
    b = rng.normal(0.6, 0.1, size=300)
    res = compare_paired(a, b, iters=3000)
    assert res.method == "paired_bootstrap"
    assert res.mcnemar is None
    assert res.diff == pytest.approx(float((a - b).mean()))


def test_compare_paired_rejects_mcnemar_on_continuous():
    with pytest.raises(ValueError):
        compare_paired([0.3, 0.7], [0.2, 0.9], method="mcnemar")


def test_paired_test_more_powerful_than_unpaired_on_correlated_data():
    """The motivating result: when two models agree on most documents and differ
    on a consistent handful, the paired test detects the difference while the
    unpaired z-test that ``model_comparator.py`` uses does not.
    """
    n = 1000
    # 150 documents both get right, 800 both get wrong: pure concordant pairs
    # that pin down the shared difficulty (positive correlation).
    both_right = [(1.0, 1.0)] * 150
    both_wrong = [(0.0, 0.0)] * 800
    # 35 discordant in A's favour, 15 in B's favour: a real, consistent edge.
    a_only = [(1.0, 0.0)] * 35
    b_only = [(0.0, 1.0)] * 15
    pairs = both_right + both_wrong + a_only + b_only
    assert len(pairs) == n
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])

    paired = compare_paired(a, b, method="mcnemar")

    # Unpaired two-sample z-test exactly as scripts/model_comparator.py computes
    # it (treating the two accuracies as independent).
    def _stderr(x):
        return float(np.std(x, ddof=1) / math.sqrt(len(x)))

    se_a, se_b = _stderr(a), _stderr(b)
    z = (a.mean() - b.mean()) / math.sqrt(se_a**2 + se_b**2)
    unpaired_p = math.erfc(abs(z) / math.sqrt(2.0))  # 2 * (1 - Phi(|z|))

    assert paired.p_value < 0.05  # paired test: significant
    assert unpaired_p > 0.05  # unpaired test: misses it
    assert paired.p_value < unpaired_p
