"""Behavioural tests for the aggregation functions in `lm_eval.api.metrics`.

These aggregations are selected by name from task YAML (`aggregation: nanmean`,
`aggregation: bleu`, ...) and turn per-document scores into the number a task
reports. A wrong aggregation does not raise: the run completes and prints a
plausible figure, so the only protection is a test that pins the arithmetic.

Coverage of `lm_eval/api/metrics.py` under the existing suite is 49% of
statements, and the aggregations below were not executed by it, despite being
referenced from shipped task configs:

    nanmean 49 configs, bleu 48, chrf 43, ter 13, weighted_perplexity 10,
    perplexity 7, f1 7, bits_per_byte 5

Each test asserts against an independently computed expected value (numpy,
sacrebleu, or arithmetic done by hand) rather than against the implementation.
"""

import math

import numpy as np
import pytest

from lm_eval.api.registry import get_aggregation


# --- central tendency -------------------------------------------------------


@pytest.mark.parametrize(
    "arr,expected",
    [
        ([1.0], 1.0),
        ([0.0, 1.0], 0.5),
        ([1, 2, 3], 2.0),
        ([0, 0, 1, 1], 0.5),
    ],
)
def test_mean_matches_numpy(arr, expected):
    assert get_aggregation("mean")(arr) == pytest.approx(expected)
    assert get_aggregation("mean")(arr) == pytest.approx(float(np.mean(arr)))


def test_nanmean_ignores_nan():
    """`nanmean` is selected by 49 task configs and must skip NaN, not propagate it."""
    arr = [1.0, float("nan"), 3.0]
    assert get_aggregation("nanmean")(arr) == pytest.approx(2.0)


def test_nanmean_all_nan_is_nan():
    """All-unscorable must stay NaN rather than collapsing to a real score."""
    assert math.isnan(get_aggregation("nanmean")([float("nan"), float("nan")]))


def test_nanmean_empty_is_nan():
    assert math.isnan(get_aggregation("nanmean")([]))


@pytest.mark.parametrize(
    "arr,expected",
    [
        ([3, 1, 2], 2.0),
        ([1.0], 1.0),
        ([5, 1, 4, 2, 3], 3.0),
    ],
)
def test_median_odd_length(arr, expected):
    """Odd-length input: the middle value of the sorted sample."""
    assert get_aggregation("median")(arr) == pytest.approx(expected)


@pytest.mark.xfail(
    reason="even-length median returns the upper middle value rather than the "
    "mean of the two middle values; fix proposed in #3668",
    strict=True,
)
@pytest.mark.parametrize("arr,expected", [([1, 2, 3, 4], 2.5), ([0, 0, 1, 1], 0.5)])
def test_median_even_length(arr, expected):
    assert get_aggregation("median")(arr) == pytest.approx(expected)


# --- weighted / perplexity family ------------------------------------------


def test_weighted_mean_divides_by_summed_weights():
    """Pairs are (numerator, weight). Expected: 10/4, not the mean of ratios."""
    items = [(4.0, 1.0), (6.0, 3.0)]
    assert get_aggregation("weighted_perplexity")  # registered
    from lm_eval.api.metrics import weighted_mean

    assert weighted_mean(items) == pytest.approx(10.0 / 4.0)


def test_perplexity_is_exp_of_negative_mean():
    items = [-1.0, -2.0]
    assert get_aggregation("perplexity")(items) == pytest.approx(math.exp(1.5))


def test_weighted_perplexity_uses_summed_weights():
    items = [(-2.0, 1.0), (-4.0, 3.0)]
    assert get_aggregation("weighted_perplexity")(items) == pytest.approx(
        math.exp(6.0 / 4.0)
    )


def test_bits_per_byte_converts_from_nats():
    """bits_per_byte is -weighted_mean / ln(2); one bit per byte is the anchor."""
    items = [(-math.log(2), 1.0)]
    assert get_aggregation("bits_per_byte")(items) == pytest.approx(1.0)


# --- classification --------------------------------------------------------


def test_f1_perfect_prediction():
    items = [(1, 1), (0, 0), (1, 1), (0, 0)]
    assert get_aggregation("f1")(items) == pytest.approx(1.0)


def test_f1_penalises_false_positives():
    """gold=[1,0,0,0], pred=[1,1,0,0]: precision 1/2, recall 1/1, F1 = 2/3."""
    items = [(1, 1), (0, 1), (0, 0), (0, 0)]
    assert get_aggregation("f1")(items) == pytest.approx(2.0 / 3.0)


def test_matthews_corrcoef_perfect_and_inverted():
    perfect = [(1, 1), (0, 0), (1, 1), (0, 0)]
    inverted = [(1, 0), (0, 1), (1, 0), (0, 1)]
    assert get_aggregation("matthews_corrcoef")(perfect) == pytest.approx(1.0)
    assert get_aggregation("matthews_corrcoef")(inverted) == pytest.approx(-1.0)


def test_brier_score_zero_when_confident_and_right():
    """Brier is lower-is-better; a confident correct prediction scores 0."""
    items = [(1, [0.0, 1.0]), (0, [1.0, 0.0])]
    assert get_aggregation("brier_score")(items) == pytest.approx(0.0)


def test_brier_score_penalises_confident_and_wrong():
    """Confidently wrong on a 2-class problem is the maximum, 2.0 per sample."""
    items = [(0, [0.0, 1.0])]
    assert get_aggregation("brier_score")(items) == pytest.approx(2.0)


# --- generation metrics ----------------------------------------------------


def test_bleu_identical_text_is_100():
    items = [("the cat sat on the mat", "the cat sat on the mat")]
    assert get_aggregation("bleu")(items) == pytest.approx(100.0)


def test_chrf_identical_text_is_100():
    items = [("the cat sat on the mat", "the cat sat on the mat")]
    assert get_aggregation("chrf")(items) == pytest.approx(100.0)


def test_ter_identical_text_is_zero():
    """TER is an error rate, so identical strings must score 0, not 100."""
    items = [("the cat sat on the mat", "the cat sat on the mat")]
    assert get_aggregation("ter")(items) == pytest.approx(0.0)


def test_bypass_is_a_sentinel_not_a_score():
    """`bypass` returns the 999 sentinel; pinned so it cannot silently become a real value."""
    assert get_aggregation("bypass")([0.0, 1.0]) == 999
