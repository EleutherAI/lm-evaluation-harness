"""Unit tests for the Scorer pipeline: construction, filtering, scoring, reduction, export/import.

Tests use real Metric, FilterEnsemble, and Instance objects with simple lambdas
to exercise the full chain without requiring heavy fixtures or model inference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pytest

from lm_eval.api._metrics.metric import Metric, take_first
from lm_eval.api.filter import FilterEnsemble
from lm_eval.api.instance import Instance
from lm_eval.api.metrics import mean
from lm_eval.scorers._base import GenScorer, LLScorer, build_scorer
from lm_eval.scorers._types import ScoredDoc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_metric(name: str = "test_metric") -> Metric:
    """Metric whose fn returns {name: [int(ref == pred) for pred in preds]}."""

    def _fn(references, predictions, **kwargs) -> dict[str, list[int]]:
        return {
            name: [
                int(r == p)
                for r, p in zip(references * len(predictions), predictions, strict=True)
            ]
        }

    return Metric(name=name, fn=_fn, aggregation=mean, reduction=take_first)


def _noop_filter_ensemble(name: str = "none") -> FilterEnsemble:
    """Build a FilterEnsemble with just the noop filter."""
    from lm_eval.filters import get_filter

    noop_cls = get_filter("noop")
    return FilterEnsemble(name=name, filters=[noop_cls])


def _make_gen_instance(
    doc_id: int,
    target: str,
    resps: list[str],
    scorer_name: str = "none",
) -> Instance:
    """Build a minimal Instance for GenScorer testing."""
    inst = Instance(
        request_type="generate_until",
        doc={},
        arguments=("prompt", {"until": ["\n"]}),
        task_name="test",
        doc_id=doc_id,
        target=target,
        resps=resps,
    )
    inst.filtered_resps[scorer_name] = resps
    return inst


def _make_ll_instances(
    doc_id: int,
    choices: list[str],
    lls: list[float],
    target: int,
    scorer_name: str = "none",
) -> list[Instance]:
    """Build instances for LLScorer testing (one per choice).

    Each instance has filtered_resps containing [(ll, is_greedy)] tuples.
    The choice with the highest ll is marked as greedy.
    """
    max_ll_idx = int(np.argmax(lls))
    instances = []
    for idx, (choice, ll) in enumerate(zip(choices, lls, strict=True)):
        inst = Instance(
            request_type="loglikelihood",
            doc={},
            arguments=("context", choice),
            task_name="test",
            doc_id=doc_id,
            idx=idx,
            target=target,
        )
        is_greedy = idx == max_ll_idx
        inst.resps = [(ll, is_greedy)]
        inst.filtered_resps[scorer_name] = [(ll, is_greedy)]
        instances.append(inst)
    return instances


def _noop_scorer(
    name: str = "none",
    metrics: list[Metric] | None = None,
) -> GenScorer:
    """Build a GenScorer with noop filter and given metrics."""
    return GenScorer(
        name=name,
        filter=_noop_filter_ensemble(name),
        metrics=metrics or [_simple_metric()],
    )


# ===========================================================================
# 1. TestBuildScorer — factory function
# ===========================================================================


class TestBuildScorer:
    """Tests for build_scorer() at _base.py:564."""

    def test_generate_until_returns_gen_scorer(self):
        scorer = build_scorer(output_type="generate_until")
        assert isinstance(scorer, GenScorer)

    def test_loglikelihood_returns_ll_scorer(self):
        scorer = build_scorer(output_type="loglikelihood")
        assert isinstance(scorer, LLScorer)

    def test_loglikelihood_rolling_returns_ll_scorer(self):
        scorer = build_scorer(output_type="loglikelihood_rolling")
        assert isinstance(scorer, LLScorer)

    def test_multiple_choice_returns_ll_scorer(self):
        scorer = build_scorer(output_type="multiple_choice")
        assert isinstance(scorer, LLScorer)

    def test_unknown_output_type_raises(self):
        with pytest.raises(ValueError, match="Cannot infer scorer"):
            build_scorer(output_type="bogus")

    def test_scorer_type_str_uses_registry(self):
        from lm_eval.scorers.extraction import FirstTokenScorer

        scorer = build_scorer(scorer_type="first_token")
        assert isinstance(scorer, FirstTokenScorer)

    def test_scorer_type_dict_passes_kwargs(self):
        from lm_eval.scorers.extraction import FirstTokenScorer

        scorer = build_scorer(scorer_type={"type": "first_token"})
        assert isinstance(scorer, FirstTokenScorer)

    def test_cfg_forwarded_to_from_dict(self):
        metric = _simple_metric("my_metric")
        scorer = build_scorer(
            cfg={"name": "strict", "metric_list": [metric]},
            output_type="generate_until",
        )
        assert scorer.name == "strict"
        assert scorer.metrics
        assert any(m.name == "my_metric" for m in scorer.metrics)

    def test_default_scorer_when_no_cfg(self):
        scorer = build_scorer(output_type="generate_until")
        assert isinstance(scorer, GenScorer)
        assert scorer.name == "none"
        # Falls back to DEFAULT_METRIC_REGISTRY for generate_until
        assert scorer.metrics
        assert any(m.name == "exact_match" for m in scorer.metrics)

    def test_cfg_metric_list_overrides_registry(self):
        scorer = build_scorer(
            cfg={"name": "test", "metric_list": [{"metric": "exact_match"}]},
            output_type="generate_until",
        )
        assert scorer.metrics
        assert len(scorer.metrics) == 1
        assert scorer.metrics[0].name == "exact_match"


# ===========================================================================
# 2. TestScorerFromDict — config resolution + precedence
# ===========================================================================


class TestScorerFromDict:
    """Tests for from_dict(), _build_filter(), _build_metrics()."""

    def test_explicit_metric_list_takes_precedence(self):
        explicit = _simple_metric("explicit")
        scorer = GenScorer.from_dict(
            {"name": "test", "metric_list": [explicit]},
            output_type="generate_until",
        )
        assert scorer.metrics
        assert len(scorer.metrics) == 1
        assert scorer.metrics[0].name == "explicit"

    def test_classvar_default_overrides_registry(self):
        @dataclass
        class CustomScorer(GenScorer):
            default_metric_cfg: ClassVar[list] = [
                {
                    "metric": "exact_match",
                    "aggregation": "mean",
                    "higher_is_better": True,
                }
            ]

        # No metric_list in cfg → class default wins over registry
        scorer = CustomScorer.from_dict({"name": "test"}, output_type="generate_until")
        assert scorer.metrics
        assert any(m.name == "exact_match" for m in scorer.metrics)

    def test_registry_fallback_when_no_metrics(self):
        scorer = GenScorer.from_dict({"name": "test"}, output_type="generate_until")
        assert scorer.metrics
        # Falls back to DEFAULT_METRIC_REGISTRY for generate_until
        assert any(m.name == "exact_match" for m in scorer.metrics)

    def test_empty_metrics_when_no_output_type(self):
        scorer = GenScorer.from_dict({"name": "test"})
        assert scorer.metrics == []

    def test_explicit_filter_takes_precedence(self):
        scorer = GenScorer.from_dict(
            {"name": "test", "filter": [{"function": "noop"}]},
        )
        assert scorer.filter.name == "test"
        assert len(scorer.filter.filters) == 1

    def test_classvar_filter_overrides_noop(self):
        @dataclass
        class CustomScorer(GenScorer):
            default_filter_cfg: ClassVar[list] = [
                {"function": "remove_whitespace"},
            ]

        scorer = CustomScorer.from_dict({"name": "test"})
        assert len(scorer.filter.filters) == 1

    def test_noop_filter_fallback(self):
        scorer = GenScorer.from_dict({"name": "test"})
        assert scorer.filter.name == "test"
        # Should have exactly one filter (noop)
        assert len(scorer.filter.filters) == 1

    def test_name_defaults_to_none(self):
        scorer = GenScorer.from_dict({})
        assert scorer.name == "none"

    def test_resolve_metrics_rejects_bad_type(self):
        with pytest.raises(TypeError, match="Metric config entries must be"):
            GenScorer._resolve_metrics([42], "generate_until")  # type:ignore[invalid-argument-type]

    def test_resolve_filters_rejects_bad_type(self):
        with pytest.raises(TypeError, match="Filter config entries must be"):
            GenScorer._resolve_filters("test", [42])  # type:ignore[invalid-argument-type]


# ===========================================================================
# 3. TestGenScorerScoring — the generation scoring chain
# ===========================================================================


class TestGenScorerScoring:
    """Tests for GenScorer.score_doc() → score() → _dispatch_metrics()."""

    def test_score_doc_returns_scored_doc(self):
        scorer = _noop_scorer()
        inst = _make_gen_instance(0, "hello", ["hello"])
        result = scorer.score_doc(0, [inst])
        assert isinstance(result, ScoredDoc)
        assert result.doc_id == 0
        assert result.reference == "hello"

    def test_score_doc_single_prediction(self):
        scorer = _noop_scorer()
        inst = _make_gen_instance(0, "hello", ["hello"])
        result = scorer.score_doc(0, [inst])
        assert "test_metric" in result.scores
        assert len(result.scores["test_metric"]) == 1
        assert result.scores["test_metric"][0] == 1

    def test_score_doc_multiple_predictions(self):
        scorer = _noop_scorer()
        inst = _make_gen_instance(0, "hello", ["hello", "world", "hello"])
        result = scorer.score_doc(0, [inst])
        assert len(result.scores["test_metric"]) == 3
        assert result.scores["test_metric"] == [1, 0, 1]

    def test_score_dispatches_to_metrics(self):
        """_dispatch_metrics calls each metric's compute()."""

        def metric_a(references, predictions) -> dict[str, list[int]]:
            return {"a": [1] * len(predictions)}

        def metric_b(references, predictions) -> dict[str, list[int]]:
            return {"b": [2] * len(predictions)}

        m_a = Metric(name="a", fn=metric_a, aggregation=mean, reduction=take_first)
        m_b = Metric(name="b", fn=metric_b, aggregation=mean, reduction=take_first)

        scorer = GenScorer(
            name="none",
            filter=_noop_filter_ensemble(),
            metrics=[m_a, m_b],
        )
        inst = _make_gen_instance(0, "ref", ["pred1", "pred2"])
        result = scorer.score_doc(0, [inst])
        assert "a" in result.scores
        assert "b" in result.scores

    def test_score_doc_missing_filtered_resps_raises(self):
        scorer = GenScorer(
            name="missing_scorer",
            filter=_noop_filter_ensemble("missing_scorer"),
            metrics=[_simple_metric()],
        )
        inst = Instance(
            request_type="generate_until",
            doc={},
            arguments=("prompt", {"until": ["\n"]}),
            task_name="test",
            doc_id=0,
            target="hello",
            resps=["hello"],
        )
        # filtered_resps is empty — scorer name not found
        with pytest.raises(KeyError, match="missing_scorer"):
            scorer.score_doc(0, [inst])

    def test_score_instances_loops_over_docs(self):
        scorer = _noop_scorer()
        inst0 = _make_gen_instance(0, "a", ["a"])
        inst1 = _make_gen_instance(1, "b", ["b"])
        results = scorer.score_instances({0: [inst0], 1: [inst1]})
        assert len(results) == 2
        assert results[0].doc_id == 0
        assert results[1].doc_id == 1

    def test_custom_score_override(self):
        """Subclass overriding score() is called correctly."""

        @dataclass
        class CustomGenScorer(GenScorer):
            def score(self, reference, predictions, **kwargs):
                return {"custom": [42] * len(predictions)}

        scorer = CustomGenScorer(
            name="none",
            filter=_noop_filter_ensemble(),
            metrics=[],
        )
        inst = _make_gen_instance(0, "ref", ["pred"])
        result = scorer.score_doc(0, [inst])
        assert result.scores == {"custom": [42]}


# ===========================================================================
# 4. TestLLScorerScoring — the log-likelihood scoring chain
# ===========================================================================


class TestLLScorerScoring:
    """Tests for LLScorer.score_doc()."""

    def _make_acc_metric(self) -> Metric:
        """Build the registered acc metric for LL tasks."""
        return Metric.from_dict(
            {"metric": "acc", "aggregation": "mean", "higher_is_better": True}
        )

    def test_score_doc_returns_scored_doc(self):
        metric = self._make_acc_metric()
        scorer = LLScorer(
            name="none",
            filter=_noop_filter_ensemble(),
            metrics=[metric],
        )
        instances = _make_ll_instances(
            doc_id=0,
            choices=["A", "B", "C"],
            lls=[-1.0, -0.5, -2.0],
            target=1,
        )
        result = scorer.score_doc(0, instances)
        assert isinstance(result, ScoredDoc)
        assert result.doc_id == 0

    def test_acc_metric_correct(self):
        """Correct choice has highest LL → acc = 1."""
        metric = self._make_acc_metric()
        scorer = LLScorer(
            name="none",
            filter=_noop_filter_ensemble(),
            metrics=[metric],
        )
        # target=1, choice 1 has highest LL (-0.5)
        instances = _make_ll_instances(
            doc_id=0,
            choices=["A", "B", "C"],
            lls=[-1.0, -0.5, -2.0],
            target=1,
        )
        result = scorer.score_doc(0, instances)
        assert result.scores["acc"] == [1]

    def test_acc_metric_incorrect(self):
        """Wrong choice has highest LL → acc = 0."""
        metric = self._make_acc_metric()
        scorer = LLScorer(
            name="none",
            filter=_noop_filter_ensemble(),
            metrics=[metric],
        )
        # target=0, but choice 1 has highest LL
        instances = _make_ll_instances(
            doc_id=0,
            choices=["A", "B", "C"],
            lls=[-1.0, -0.5, -2.0],
            target=0,
        )
        result = scorer.score_doc(0, instances)
        assert result.scores["acc"] == [0]

    def test_scores_wrapped_in_list(self):
        """All metric values are [scalar] (single-element list)."""
        metric = self._make_acc_metric()
        scorer = LLScorer(
            name="none",
            filter=_noop_filter_ensemble(),
            metrics=[metric],
        )
        instances = _make_ll_instances(
            doc_id=0,
            choices=["A", "B"],
            lls=[-1.0, -0.5],
            target=1,
        )
        result = scorer.score_doc(0, instances)
        for values in result.scores.values():
            assert isinstance(values, list)
            assert len(values) == 1


# ===========================================================================
# 5. TestScorerReduce — reduction step
# ===========================================================================


class TestScorerReduce:
    """Tests for reduce() and set_results()."""

    def test_single_value_no_reduction_needed(self):
        scorer = _noop_scorer()
        scored_docs = {
            0: ScoredDoc(doc_id=0, reference="ref", scores={"test_metric": [0.5]}),
        }
        scorer.reduce(scored_docs)
        assert scored_docs[0].reduced_scores["test_metric"] == 0.5

    def test_multi_value_uses_metric_reduction(self):
        def _mean_reduction(references, values):
            return sum(values) / len(values)

        metric = Metric(
            name="test_metric",
            fn=lambda refs, preds: {"test_metric": [1]},
            aggregation=mean,
            reduction=_mean_reduction,
        )
        scorer = GenScorer(
            name="none",
            filter=_noop_filter_ensemble(),
            metrics=[metric],
        )
        scored_docs = {
            0: ScoredDoc(doc_id=0, reference="ref", scores={"test_metric": [1, 0, 1]}),
        }
        scorer.reduce(scored_docs)
        assert scored_docs[0].reduced_scores["test_metric"] == pytest.approx(2 / 3)

    def test_dict_reduction_creates_composite_keys(self):
        """Reduction returning {"pass@1": 1.0} → "pass@1(metric)" key."""

        def _dict_reduction(references, values):
            return {"pass@1": 1.0, "pass@3": 0.8}

        metric = Metric(
            name="metric",
            fn=lambda refs, preds: {"metric": [1]},
            aggregation=mean,
            reduction=_dict_reduction,
        )
        scorer = GenScorer(
            name="none",
            filter=_noop_filter_ensemble(),
            metrics=[metric],
        )
        scored_docs = {
            0: ScoredDoc(doc_id=0, reference="ref", scores={"metric": [1, 0, 1]}),
        }
        scorer.reduce(scored_docs)
        assert "pass@1(metric)" in scored_docs[0].reduced_scores
        assert "pass@3(metric)" in scored_docs[0].reduced_scores
        assert scored_docs[0].reduced_scores["pass@1(metric)"] == 1.0
        assert scored_docs[0].reduced_scores["pass@3(metric)"] == 0.8

    def test_missing_reduction_warns_and_takes_first(self, caplog):
        _metric = Metric(
            name="test_metric",
            fn=lambda refs, preds: {"test_metric": [1]},
            aggregation=mean,
            reduction=None,
        )
        # Metric.__post_init__ forces reduction=take_first when None, so
        # we need a scorer with no matching metric for the scored metric name
        scorer = GenScorer(
            name="none",
            filter=_noop_filter_ensemble(),
            metrics=[],  # No metrics → no reduction function found
        )
        scored_docs = {
            0: ScoredDoc(
                doc_id=0, reference="ref", scores={"unknown_metric": [10, 20, 30]}
            ),
        }
        with caplog.at_level(logging.WARNING, logger="lm_eval.scorers._base"):
            scorer.reduce(scored_docs)
        assert scored_docs[0].reduced_scores["unknown_metric"] == 10
        assert "No reduction function" in caplog.text

    def test_set_results_stores_and_reduces(self):
        scorer = _noop_scorer()
        scored_docs = {
            0: ScoredDoc(doc_id=0, reference="ref", scores={"test_metric": [0.7]}),
            1: ScoredDoc(doc_id=1, reference="ref2", scores={"test_metric": [0.3]}),
        }
        scorer.set_results(scored_docs)
        assert scorer._scored_docs is scored_docs
        assert scored_docs[0].reduced_scores["test_metric"] == 0.7
        assert scored_docs[1].reduced_scores["test_metric"] == 0.3


# ===========================================================================
# 6. TestScorerExportImport — distributed serialization round-trip
# ===========================================================================


class TestScorerExportImport:
    """Tests for export_reduced() and import_reduced()."""

    def _populated_scorer(self) -> GenScorer:
        scorer = _noop_scorer()
        scored_docs = {
            0: ScoredDoc(
                doc_id=0,
                reference="ref0",
                scores={"test_metric": [1.0]},
                reduced_scores={"test_metric": 1.0},
            ),
            1: ScoredDoc(
                doc_id=1,
                reference="ref1",
                scores={"test_metric": [0.0]},
                reduced_scores={"test_metric": 0.0},
            ),
        }
        scorer._scored_docs = scored_docs
        return scorer

    def test_export_reduced_flat_structure(self):
        scorer = self._populated_scorer()
        exported = scorer.export_reduced()
        assert "test_metric" in exported
        assert exported["test_metric"] == [1.0, 0.0]

    def test_import_reduced_rebuilds_scored_docs(self):
        scorer = _noop_scorer()
        scorer.import_reduced({"test_metric": [1.0, 0.0]})
        assert len(scorer._scored_docs) == 2
        assert scorer._scored_docs[0].reduced_scores["test_metric"] == 1.0
        assert scorer._scored_docs[1].reduced_scores["test_metric"] == 0.0

    def test_export_import_roundtrip(self):
        scorer = self._populated_scorer()
        exported = scorer.export_reduced()

        new_scorer = _noop_scorer()
        new_scorer.import_reduced(exported)

        re_exported = new_scorer.export_reduced()
        assert re_exported == exported

    def test_import_reduced_sets_reference_to_none(self):
        scorer = _noop_scorer()
        scorer.import_reduced({"test_metric": [1.0, 0.0]})
        for sd in scorer._scored_docs.values():
            assert sd.reference is None


# ===========================================================================
# 7. TestApplyFilter — filter application
# ===========================================================================


class TestApplyFilter:
    """Tests for apply_filter()."""

    def test_apply_filter_populates_filtered_resps(self):
        scorer = _noop_scorer(name="my_filter")
        inst = Instance(
            request_type="generate_until",
            doc={"text": "hello"},
            arguments=("prompt", {"until": ["\n"]}),
            task_name="test",
            doc_id=0,
            target="target",
            resps=["response1", "response2"],
        )
        scorer.apply_filter([inst])
        assert "my_filter" in inst.filtered_resps
        assert inst.filtered_resps["my_filter"] == ["response1", "response2"]

    def test_noop_filter_preserves_responses(self):
        scorer = _noop_scorer()
        resps = ["alpha", "beta", "gamma"]
        inst = Instance(
            request_type="generate_until",
            doc={},
            arguments=("prompt", {"until": ["\n"]}),
            task_name="test",
            doc_id=0,
            target="target",
            resps=resps,
        )
        scorer.apply_filter([inst])
        # noop filter should pass through responses unchanged
        assert list(inst.filtered_resps["none"]) == resps
