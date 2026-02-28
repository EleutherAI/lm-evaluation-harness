"""Tests for the metrics in lm_eval/api/metrics/.

Tests cover:
  - Per-sample metric functions (acc, acc_norm, acc_mutual_info, brier_score, etc.)
  - LLResults construction and mutual-info slicing
  - Aggregation functions (mean, median, perplexity)
  - Bootstrap stderr helpers
  - Corpus-level metrics (Perplexity, F1, AccAll)
  - exact_match for generate_until
"""

import math
import unittest.mock as mock

import numpy as np
import pytest

from lm_eval.api.metrics.aggregations import mean, median, perplexity
from lm_eval.api.metrics.generation import exact_match_fn
from lm_eval.api.metrics.ll import (
    _softmax as softmax,
    acc,
    acc_mutual_info_fn,
    acc_norm,
    bpb,
    brier_score,
    choice_logprob,
    logprob_fn,
)
from lm_eval.api.metrics.results import LLResults
from lm_eval.api.metrics.stderr import (
    _bootstrap_internal_no_mp,
    mean_stderr,
    sample_stddev,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ll_results(
    lls: list[float],
    is_greedy: list[bool] | None = None,
    targets: int | list[int] = 0,
    choices: list[str] | None = None,
    lls_mutual_info: list[float] | None = None,
) -> LLResults:
    """Build an LLResults for unit tests without needing Instance objects."""
    if is_greedy is None:
        is_greedy = [False] * len(lls)
    if choices is None:
        choices = [chr(65 + i) for i in range(len(lls))]  # A, B, C, ...
    kwargs = {}
    if lls_mutual_info is not None:
        kwargs["lls_mutual_info"] = np.array(lls_mutual_info)
    return LLResults(
        results=[],
        lls=np.array(lls),
        is_greedy=is_greedy,
        targets=targets,
        choices=choices,
        **kwargs,
    )


# ===========================================================================
# Per-sample metric functions
# ===========================================================================


class TestAcc:
    def test_acc_correct(self):
        pred = _make_ll_results([-2.0, -1.0, -3.0], targets=1)
        assert acc(1, pred) == 1

    def test_acc_wrong(self):
        pred = _make_ll_results([-1.0, -2.0, -3.0], targets=1)
        # argmax is 0 but gold is 1
        assert acc(1, pred) == 0

    def test_acc_single_ll_greedy(self):
        """Single loglikelihood: acc = greedy decode match."""
        pred = _make_ll_results([-1.5], is_greedy=[True], choices=["x"])
        assert acc(0, pred) == 1

    def test_acc_single_ll_not_greedy(self):
        pred = _make_ll_results([-1.5], is_greedy=[False], choices=["x"])
        assert acc(0, pred) == 0

    def test_acc_multiple_targets(self):
        pred = _make_ll_results([-2.0, -1.0, -3.0], targets=[0, 1])
        # argmax=1, which is in [0, 1]
        assert acc([0, 1], pred, multiple_targets=True) == 1

    def test_acc_multiple_targets_miss(self):
        pred = _make_ll_results([-2.0, -1.0, -3.0], targets=[0, 2])
        # argmax=1, which is not in [0, 2]
        assert acc([0, 2], pred, multiple_targets=True) == 0

    def test_acc_multiple_targets_ignores_minus100(self):
        pred = _make_ll_results([-2.0, -1.0, -3.0], targets=[1, -100])
        # argmax=1, -100 is filtered out, so [1] matches
        assert acc([1, -100], pred, multiple_targets=True) == 1


class TestAccNorm:
    def test_acc_norm_normalizes_by_length(self):
        # lls:       [-2.0, -1.0, -3.0]
        # choices:   ["AB", "B", "CDE"]   -> char_len = [2, 1, 3]
        # ll/len:    [-1.0, -1.0, -1.0]   -> tie, argmax = 0
        pred = _make_ll_results([-2.0, -1.0, -3.0], choices=["AB", "B", "CDE"])
        result = acc_norm(0, pred)
        assert result == 1  # argmax of normalized is 0

    def test_acc_norm_picks_shorter_choice(self):
        # lls:       [-2.0, -2.0]
        # choices:   ["A", "AB"]   -> char_len = [1, 2]
        # ll/len:    [-2.0, -1.0]  -> argmax = 1
        pred = _make_ll_results([-2.0, -2.0], choices=["A", "AB"])
        assert acc_norm(1, pred) == 1


class TestAccMutualInfo:
    def test_basic_mutual_info(self):
        """acc_mutual_info picks argmax of lls_mutual_info, not raw lls."""
        pred = _make_ll_results(
            lls=[-1.0, -2.0, -3.0],
            targets=1,
            lls_mutual_info=[
                -0.5,  # A: conditional - unconditional
                0.0,  # B: wins with mutual info
                0.0,  # C
            ],
        )
        # argmax(lls_mutual_info) = 1 (or 2, tie-break), gold = 1
        assert acc_mutual_info_fn(1, pred) == 1

    def test_mutual_info_differs_from_raw_acc(self):
        """When conditional and mutual-info predictions diverge."""
        pred = _make_ll_results(
            lls=[-1.0, -2.0, -3.0],  # A wins conditionally
            targets=1,
            lls_mutual_info=[
                -0.5,  # A: -1.0 - (-0.5)
                0.0,  # B: -2.0 - (-2.0) -> wins with MI
                0.0,  # C: -3.0 - (-3.0)
            ],
        )
        # Regular acc picks A (index 0), gold is 1 -> wrong
        assert acc(1, pred) == 0
        # Mutual info picks B (index 1), gold is 1 -> correct
        assert acc_mutual_info_fn(1, pred) == 1


class TestBrierScore:
    def test_perfect_prediction(self):
        """Brier score = 0 when model puts all mass on correct answer."""
        # lls such that softmax gives ~[0, 1, 0]
        pred = _make_ll_results([-100.0, 0.0, -100.0], targets=1)
        score = brier_score(1, pred)
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_uniform_prediction(self):
        """Brier score for uniform prediction over 3 choices, gold = 0."""
        pred = _make_ll_results([0.0, 0.0, 0.0], targets=0)
        score = brier_score(0, pred)
        # softmax([0,0,0]) = [1/3, 1/3, 1/3]
        # one_hot = [1, 0, 0]
        # sum((1/3-1)^2 + (1/3-0)^2 + (1/3-0)^2) = (2/3)^2 + 2*(1/3)^2 = 4/9 + 2/9 = 6/9
        assert score == pytest.approx(6 / 9, abs=1e-6)


class TestBpb:
    def test_bpb_basic(self):
        pred = _make_ll_results([-2.0, -1.0], choices=["ab", "c"], targets=0)
        # bpb = -lls[0] / byte_len[0] * NAT_TO_BIT
        # byte_len("ab") = 2
        # bpb = 2.0 / 2 * (1/ln2) = 1/ln2 ≈ 1.4427
        result = bpb(0, pred)
        assert result == pytest.approx(1.0 / np.log(2.0), rel=1e-6)


class TestLogprob:
    def test_logprob_returns_gold_ll(self):
        pred = _make_ll_results([-2.0, -1.0, -3.0], targets=1)
        assert logprob_fn(1, pred) == pytest.approx(-1.0)


class TestChoiceLogprob:
    def test_basic(self):
        pred = _make_ll_results([-2.0, -1.0, -3.0], targets=1)
        # log(softmax) at index 1 = lls[1] - logsumexp(lls)
        lls = np.array([-2.0, -1.0, -3.0])
        expected = float(lls[1] - np.logaddexp.reduce(lls))
        assert choice_logprob(1, pred) == pytest.approx(expected)


# ===========================================================================
# LLResults construction
# ===========================================================================


class TestLLResultsDefaults:
    def test_default_lls_mutual_info_is_empty_array(self):
        """Regression: default_factory must call empty_array, not store the function."""
        pred = _make_ll_results([-1.0, -2.0], targets=0)
        assert isinstance(pred.lls_mutual_info, np.ndarray)
        assert len(pred.lls_mutual_info) == 0

    def test_char_len(self):
        pred = _make_ll_results([-1.0, -2.0], choices=["AB", "CDE"])
        np.testing.assert_array_equal(pred.char_len(), [2.0, 3.0])

    def test_byte_len(self):
        pred = _make_ll_results([-1.0], choices=["hello"])
        np.testing.assert_array_equal(pred.byte_len(), [5.0])

    def test_char_len_no_choices(self):
        """When choices is empty, char_len returns ones matching lls length."""
        pred = LLResults(
            results=[],
            lls=np.array([-1.0, -2.0, -3.0]),
            is_greedy=[False, False, False],
            targets=0,
            choices=[],
        )
        np.testing.assert_array_equal(pred.char_len(), [1.0, 1.0, 1.0])


# ===========================================================================
# exact_match (generate_until)
# ===========================================================================


class TestExactMatch:
    def test_perfect_match(self):
        result = exact_match_fn(
            references=["hello", "world"], predictions=["hello", "world"]
        )
        assert result["exact_match"] == [1, 1]

    def test_no_match(self):
        result = exact_match_fn(references=["hello"], predictions=["world"])
        assert result["exact_match"] == [0]

    def test_partial_match(self):
        result = exact_match_fn(references=["a", "b"], predictions=["a", "c"])
        assert result["exact_match"] == [1, 0]

    def test_ignore_case(self):
        result = exact_match_fn(
            references=["Hello"], predictions=["hello"], ignore_case=True
        )
        assert result["exact_match"] == [1]

    def test_ignore_punctuation(self):
        result = exact_match_fn(
            references=["hello!"], predictions=["hello"], ignore_punctuation=True
        )
        assert result["exact_match"] == [1]

    def test_regex_ignore(self):
        result = exact_match_fn(
            references=["answer: 42"],
            predictions=["answer: 42!"],
            regexes_to_ignore=[r"[!]"],
        )
        assert result["exact_match"] == [1]

    def test_multiple_targets_match(self):
        result = exact_match_fn(
            references=[["a", "b"]], predictions=["b"], multiple_targets=True
        )
        assert result["exact_match"] == [1]

    def test_multiple_targets_miss(self):
        result = exact_match_fn(
            references=[["a", "b"]], predictions=["c"], multiple_targets=True
        )
        assert result["exact_match"] == [0]

    def test_multiple_targets_with_repeats(self):
        result = exact_match_fn(
            references=[["a", "b"]], predictions=["a", "c", "b"], multiple_targets=True
        )
        assert result["exact_match"] == [1, 0, 1]

    def test_multiple_targets_ignore_case(self):
        result = exact_match_fn(
            references=[["Hello", "Hi"]],
            predictions=["hi"],
            multiple_targets=True,
            ignore_case=True,
        )
        assert result["exact_match"] == [1]

    def test_multiple_targets_single_ref_fallback(self):
        result = exact_match_fn(
            references=["a"], predictions=["a"], multiple_targets=True
        )
        assert result["exact_match"] == [1]


# ===========================================================================
# Aggregation functions
# ===========================================================================


class TestAggregations:
    def test_mean(self):
        assert mean([1, 2, 3, 4, 5]) == 3.0

    def test_median_odd(self):
        assert median([3, 1, 2]) == 2  # sorted: [1,2,3], middle = 2

    def test_median_even(self):
        assert median([4, 1, 3, 2]) == 3  # sorted: [1,2,3,4], index 2 = 3

    def test_median_already_sorted(self):
        assert median([1, 2, 3, 4, 5]) == 3

    def test_median_unsorted(self):
        """Regression: median must sort before indexing."""
        assert median([5, 1, 3]) == 3  # not 1 (middle of unsorted)

    def test_perplexity(self):
        items = [-1.0, -2.0, -3.0]
        expected = math.exp(-mean(items))
        assert perplexity(items) == pytest.approx(expected)


# ===========================================================================
# Softmax utility
# ===========================================================================


class TestSoftmax:
    def test_sums_to_one(self):
        result = softmax(np.array([1.0, 2.0, 3.0]))
        assert result.sum() == pytest.approx(1.0)

    def test_uniform(self):
        result = softmax(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(result, [1 / 3, 1 / 3, 1 / 3])

    def test_extreme_values(self):
        """Softmax should be numerically stable with large values."""
        result = softmax(np.array([1000.0, 1000.0]))
        np.testing.assert_allclose(result, [0.5, 0.5])


# ===========================================================================
# Bootstrap / stderr
# ===========================================================================


class TestBootstrap:
    def test_bootstrap_internal_no_mp(self):
        data = [1, 2, 3, 4, 5]

        with (
            mock.patch("tqdm.tqdm", side_effect=lambda x, **kw: x),
            mock.patch("builtins.print"),
        ):
            result = _bootstrap_internal_no_mp(mean, data, 1000)

        assert len(result) == 1000
        assert all(isinstance(x, (int, float)) for x in result)
        # Bootstrap mean should be close to the original mean
        assert abs(mean(result) - mean(data)) < 0.5

    def test_mean_stderr(self):
        data = [1, 2, 3, 4, 5]
        se = mean_stderr(data)
        # stderr = stddev / sqrt(n) = sqrt(2.5) / sqrt(5) ≈ 0.7071
        expected = sample_stddev(data) / math.sqrt(len(data))
        assert se == pytest.approx(expected)


# ===========================================================================
# Corpus-level metrics
# ===========================================================================


class TestCorpusPerplexity:
    def test_corpus_perplexity(self):
        from lm_eval.api.metrics.corpus import Perplexity

        ppl = Perplexity()
        # Per-doc: extract gold ll
        pred1 = _make_ll_results([-2.0, -1.0], targets=1)
        pred2 = _make_ll_results([-3.0, -0.5], targets=1)
        item1 = ppl(1, pred1)  # -1.0
        item2 = ppl(1, pred2)  # -0.5
        assert item1 == pytest.approx(-1.0)
        assert item2 == pytest.approx(-0.5)
        # Aggregation: exp(-mean(lls))
        result = ppl.aggregation([item1, item2])
        assert result == pytest.approx(math.exp(-mean([-1.0, -0.5])))

    def test_single_ll_perplexity(self):
        from lm_eval.api.metrics.corpus import Perplexity

        ppl = Perplexity()
        pred = _make_ll_results([-2.5], choices=["x"])
        assert ppl(0, pred) == pytest.approx(-2.5)


class TestSacreformat:
    def test_single_ref_strings(self):
        """Single reference per sample, flat list of strings."""
        from lm_eval.api.metrics.corpus import _sacreformat

        refs = ["The cat sat.", "It was raining."]
        preds = ["A cat sat.", "It rained."]
        out_refs, out_preds = _sacreformat(refs, preds)
        # sacrebleu expects refs as List[List[str]] where inner list = one per sample
        assert out_refs == [("The cat sat.", "It was raining.")]
        assert out_preds == ["A cat sat.", "It rained."]

    def test_multiple_refs_per_sample(self):
        """Two references per sample, transposed to per-stream grouping."""
        from lm_eval.api.metrics.corpus import _sacreformat

        refs = [["ref1a", "ref1b"], ["ref2a", "ref2b"]]
        preds = ["pred1", "pred2"]
        out_refs, out_preds = _sacreformat(refs, preds)
        # Transposed: stream 0 = [ref1a, ref2a], stream 1 = [ref1b, ref2b]
        assert out_refs == [("ref1a", "ref2a"), ("ref1b", "ref2b")]
        assert out_preds == ["pred1", "pred2"]

    def test_nested_preds_unwrapped(self):
        """Predictions wrapped in single-element lists get unwrapped."""
        from lm_eval.api.metrics.corpus import _sacreformat

        refs = ["ref1", "ref2"]
        preds = [["pred1"], ["pred2"]]
        out_refs, out_preds = _sacreformat(refs, preds)
        assert out_preds == ["pred1", "pred2"]


class TestCorpusMetricReduce:
    """Tests for the reduce step that sits between __call__ and aggregation.

    __call__ is invoked once per sample.  With repeats, predictions is a
    list[str] with one string per repeat while references (list[str]) stays
    the same.  reduce must strip the extra repeat predictions so that
    aggregation / _sacreformat see exactly one prediction per sample.
    """

    def test_reduce_single_repeat_passes_through(self):
        from lm_eval.api.metrics.corpus import Bleu

        bleu = Bleu()
        # repeat=1: predictions has exactly one string
        item = bleu(["The cat."], ["A cat."])
        reduced = bleu.reduce(["The cat."], [item])
        assert reduced == item

    def test_reduce_strips_extra_repeat_predictions(self):
        """With repeat>1, predictions has multiple strings; reduce keeps only the first."""
        from lm_eval.api.metrics.corpus import Bleu

        bleu = Bleu()
        refs = ["The cat."]
        # __call__ once per sample: 3 repeats bundled in predictions
        item = bleu(refs, ["A cat.", "The dog.", "A feline."])
        reduced = bleu.reduce(refs, [item])
        # Must strip down to first repeat only
        assert reduced == (refs, ["A cat."])

    def test_reduce_strips_repeats_all_metrics(self):
        """All sacrebleu metrics strip extra repeat predictions via reduce."""
        from lm_eval.api.metrics.corpus import Bleu, Chrf, Ter

        refs = ["The cat sat on the mat."]
        for cls in (Bleu, Chrf, Ter):
            metric = cls()
            # 3 repeats in one __call__
            item = metric(refs, ["pred1", "pred2", "pred3"])
            reduced = metric.reduce(refs, [item])
            assert reduced == (refs, ["pred1"])

    def test_full_pipeline_with_repeats(self):
        """End-to-end: repeat=3 across two samples, reduce → aggregation."""
        from lm_eval.api.metrics.corpus import Bleu

        bleu = Bleu()

        # Sample 1: 3 repeats, first is perfect
        s1_refs = ["The cat sat on the mat."]
        s1_raw = bleu(s1_refs, ["The cat sat on the mat.", "A cat.", "Cats."])
        s1_reduced = bleu.reduce(s1_refs, [s1_raw])

        # Sample 2: 3 repeats, first is perfect
        s2_refs = ["It was a fine day."]
        s2_raw = bleu(s2_refs, ["It was a fine day.", "Nice day.", "Good."])
        s2_reduced = bleu.reduce(s2_refs, [s2_raw])

        # reduce kept only the first (perfect) repeat, so BLEU = 100
        score = bleu.aggregation([s1_reduced, s2_reduced])
        assert score == pytest.approx(100.0)

    def test_reduce_single_repeat_unchanged(self):
        """With repeat=1, reduce returns the result unchanged."""
        from lm_eval.api.metrics.corpus import Bleu, Chrf, Ter

        for cls in (Bleu, Chrf, Ter):
            metric = cls()
            item = metric(["hello"], ["hello"])
            assert metric.reduce(["hello"], [item]) == item


class TestBleu:
    def test_call_returns_refs_and_preds(self):
        from lm_eval.api.metrics.corpus import Bleu

        bleu = Bleu()
        result = bleu(["the cat"], ["a cat"])
        assert result == (["the cat"], ["a cat"])

    def test_full_pipeline_call_reduce_aggregate(self):
        """End-to-end: __call__ -> reduce -> aggregation."""
        from lm_eval.api.metrics.corpus import Bleu

        bleu = Bleu()
        # __call__ per document
        raw1 = bleu(["The cat sat on the mat."], ["The cat sat on the mat."])
        raw2 = bleu(["It was a fine day."], ["It was a fine day."])
        # reduce (simulating repeat=1)
        item1 = bleu.reduce(["The cat sat on the mat."], [raw1])
        item2 = bleu.reduce(["It was a fine day."], [raw2])
        # aggregation
        score = bleu.aggregation([item1, item2])
        assert score == pytest.approx(100.0)

    def test_bleu_zero_for_no_overlap(self):
        from lm_eval.api.metrics.corpus import Bleu

        bleu = Bleu()
        raw = bleu(["aaa bbb ccc ddd"], ["xxx yyy zzz www"])
        item = bleu.reduce(["aaa bbb ccc ddd"], [raw])
        score = bleu.aggregation([item])
        assert score == pytest.approx(0.0)

    def test_bleu_partial_match(self):
        from lm_eval.api.metrics.corpus import Bleu

        bleu = Bleu()
        raw = bleu(["The cat sat on the mat."], ["The cat lay on the mat."])
        item = bleu.reduce(["The cat sat on the mat."], [raw])
        score = bleu.aggregation([item])
        assert 0.0 < score < 100.0


class TestChrf:
    def test_call_returns_refs_and_preds(self):
        from lm_eval.api.metrics.corpus import Chrf

        chrf = Chrf()
        result = chrf(["the cat"], ["a cat"])
        assert result == (["the cat"], ["a cat"])

    def test_full_pipeline_call_reduce_aggregate(self):
        """End-to-end: __call__ -> reduce -> aggregation."""
        from lm_eval.api.metrics.corpus import Chrf

        chrf = Chrf()
        raw1 = chrf(["The cat sat on the mat."], ["The cat sat on the mat."])
        raw2 = chrf(["It was a fine day."], ["It was a fine day."])
        item1 = chrf.reduce(["The cat sat on the mat."], [raw1])
        item2 = chrf.reduce(["It was a fine day."], [raw2])
        score = chrf.aggregation([item1, item2])
        assert score == pytest.approx(100.0)

    def test_chrf_partial_match(self):
        from lm_eval.api.metrics.corpus import Chrf

        chrf = Chrf()
        raw = chrf(["The cat sat on the mat."], ["The cat lay on the mat."])
        item = chrf.reduce(["The cat sat on the mat."], [raw])
        score = chrf.aggregation([item])
        assert 0.0 < score < 100.0


class TestTer:
    def test_call_returns_refs_and_preds(self):
        from lm_eval.api.metrics.corpus import Ter

        ter = Ter()
        result = ter(["the cat"], ["a cat"])
        assert result == (["the cat"], ["a cat"])

    def test_full_pipeline_call_reduce_aggregate(self):
        """End-to-end: __call__ -> reduce -> aggregation."""
        from lm_eval.api.metrics.corpus import Ter

        ter = Ter()
        raw1 = ter(["The cat sat on the mat."], ["The cat sat on the mat."])
        raw2 = ter(["It was a fine day."], ["It was a fine day."])
        item1 = ter.reduce(["The cat sat on the mat."], [raw1])
        item2 = ter.reduce(["It was a fine day."], [raw2])
        score = ter.aggregation([item1, item2])
        assert score == pytest.approx(0.0)

    def test_ter_nonzero_for_mismatch(self):
        from lm_eval.api.metrics.corpus import Ter

        ter = Ter()
        raw = ter(["The cat sat on the mat."], ["A dog lay on the rug."])
        item = ter.reduce(["The cat sat on the mat."], [raw])
        score = ter.aggregation([item])
        assert score > 0.0
