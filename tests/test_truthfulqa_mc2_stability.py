from decimal import Decimal, localcontext
from itertools import permutations

import numpy as np
import pytest

from lm_eval.tasks.truthfulqa.utils import process_results_mc2


def _score(log_likelihoods, labels):
    doc = {"mc2_targets": {"labels": labels}}
    results = [(score, False) for score in log_likelihoods]
    return process_results_mc2(doc, results)["acc"]


def _decimal_reference(log_likelihoods, labels):
    with localcontext() as ctx:
        ctx.prec = 80
        probabilities = [Decimal(str(float(score))).exp() for score in log_likelihoods]
        correct = sum(
            p for p, label in zip(probabilities, labels, strict=True) if label == 1
        )
        return float(correct / sum(probabilities))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("offset", [0.0, -1000.0, -10000.0])
def test_mc2_preserves_relative_probability_for_small_likelihoods(dtype, offset):
    scores = np.array([-1.0, -2.0, -3.0], dtype=dtype) + dtype(offset)
    labels = [1, 0, 1]
    with np.errstate(invalid="raise", divide="raise"):
        actual = _score(scores, labels)
    assert np.isfinite(actual)
    assert actual == pytest.approx(_decimal_reference(scores, labels), rel=1e-6)


@pytest.mark.parametrize("labels", [[1, 0, 0], [1, 1, 0], [1, 1, 1]])
def test_mc2_equal_small_likelihoods(labels):
    assert _score([-1000.0] * 3, labels) == pytest.approx(sum(labels) / 3)


@pytest.mark.parametrize("offset", [-1000.0, -10000.0])
def test_mc2_is_invariant_to_common_log_likelihood_shift(offset):
    scores = np.array([-1.0, -2.0, -3.0, -4.0])
    assert _score(scores + offset, [1, 0, 1, 0]) == pytest.approx(
        _score(scores, [1, 0, 1, 0])
    )


def test_mc2_permuting_answers_and_labels_preserves_score():
    scores = [-1000.0, -1001.0, -1002.0]
    labels = [1, 0, 1]
    expected = _decimal_reference(scores, labels)
    for order in permutations(range(3)):
        actual = _score([scores[i] for i in order], [labels[i] for i in order])
        assert actual == pytest.approx(expected)


@pytest.mark.parametrize("labels, expected", [([1, 0], 1.0), ([0, 1], 0.0)])
def test_mc2_handles_zero_probability_option(labels, expected):
    assert _score([-1000.0, -np.inf], labels) == expected


def test_mc2_does_not_turn_undefined_distribution_into_valid_score():
    # All-impossible candidates have no normalizable mass, before or after this fix.
    with np.errstate(invalid="ignore", divide="ignore"):
        assert np.isnan(_score([-np.inf, -np.inf], [1, 0]))


def test_mc2_matches_existing_formula_in_nondegenerate_range():
    rng = np.random.default_rng(42)
    for _ in range(100):
        scores = rng.uniform(-50.0, -1.0, size=8)
        labels = rng.integers(0, 2, size=8)
        probabilities = np.exp(scores)
        expected = probabilities[labels == 1].sum() / probabilities.sum()
        assert _score(scores, labels) == pytest.approx(expected, rel=1e-12, abs=1e-14)
