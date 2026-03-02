# type:ignore[invalid-assignment]
"""End-to-end tests for the aggregation pipeline: raw per-sample values → task aggregation → group aggregation.

Tests the full path through:
  _compute_task_aggregations → _collect_results → _aggregate_groups
  (wrapped by _process_results)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import pytest

from lm_eval.api.filter import FilterEnsemble
from lm_eval.api.group import Group
from lm_eval.api.metrics import mean
from lm_eval.config.group import AggMetricConfig
from lm_eval.evaluator_utils import (
    _process_results,
)
from lm_eval.scorers import ScoredDoc, Scorer


if TYPE_CHECKING:
    from lm_eval.api.task import Task
    from lm_eval.evaluator_utils import (
        _ResultAcc,
    )
    from lm_eval.result_schema import _TaskMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _m(d: dict[str, Any]) -> _TaskMetrics:
    """Cast a plain dict to _TaskMetrics for tests."""
    return d  # type: ignore[return-value]


def _reduced_docs_from_flat(
    metrics_dict: dict[str, list],
) -> dict[int, dict[str, float]]:
    """Build _reduced_docs from flat {metric: [values]} for testing."""
    n_docs = max((len(v) for v in metrics_dict.values()), default=0)
    docs: dict[int, dict[str, float]] = {}
    for i in range(n_docs):
        docs[i] = {mn: vals[i] for mn, vals in metrics_dict.items() if i < len(vals)}
    return docs


def _build_multi_scorer_scorers(
    raw_metrics: dict[tuple[str, str], list],
    agg: dict[str, Any] | None = None,
    hib: dict[str, bool] | None = None,
) -> list[Scorer]:
    """Build Scorer objects from tuple-keyed raw_metrics with _reduced_docs populated."""
    from lm_eval.api.metrics import Metric

    agg = agg or {}
    hib = hib or {}

    scorers_data: dict[str, dict[str, list]] = defaultdict(dict)
    for (metric_name, scorer_name), values in raw_metrics.items():
        scorers_data[scorer_name][metric_name] = values

    scorers = []
    noop_filter = FilterEnsemble("_unused", [("identity", None)])
    for scorer_name, metrics_dict in scorers_data.items():
        metrics = []
        for metric_name in metrics_dict:
            agg_fn = agg.get(metric_name, mean)
            metrics.append(
                Metric(
                    name=metric_name,
                    fn=lambda *a, **kw: 0,
                    aggregation=agg_fn,
                    higher_is_better=hib.get(metric_name, True),
                )
            )
        scorer = Scorer(
            name=scorer_name,
            filter=noop_filter,
            metrics=metrics,
        )
        scorer._reduced_docs = _reduced_docs_from_flat(metrics_dict)
        scorers.append(scorer)
    return scorers


def _make_acc(
    task: Task,
    raw_metrics: dict[tuple[str, str], list],
    agg: dict[str, Any] | None = None,
    hib: dict[str, bool] | None = None,
) -> _ResultAcc:
    """Build _ResultAcc and populate task scorers with scored_docs."""
    task._scorers = _build_multi_scorer_scorers(
        raw_metrics, agg=agg or {}, hib=hib or {}
    )
    return {
        "task": task,
        "logged_samples": [],
    }


# ---------------------------------------------------------------------------
# Pipeline: task raw values → _collect_results → group aggregation
# ---------------------------------------------------------------------------


class TestTaskToGroupPipeline:
    """End-to-end: raw per-sample values flow through task agg into group agg."""

    def test_single_task_single_group(self, make_task):
        """One task, one group — group metric == task metric."""
        task = make_task("t1", n_eval_docs=4)
        group = Group(name="g", aggregate_metric_list=[AggMetricConfig(metric="acc")])
        group.add(task)

        agg = {"acc": mean}
        acc = {"t1": _make_acc(task, {("acc", "none"): [1.0, 0.0, 1.0, 0.0]}, agg=agg)}
        result = _process_results(acc, groups={"g": group}, bootstrap_iters=0)

        # Task metrics
        assert result.metrics["t1"]["acc,none"] == pytest.approx(0.5)
        assert result.metrics["t1"]["sample_len"] == 4

        # Group metrics should match (single task)
        assert result.metrics["g"]["acc,none"] == pytest.approx(0.5)
        assert result.metrics["g"]["sample_len"] == 4

    def test_two_tasks_weighted_group(self, make_task):
        """Two tasks with different sizes — group uses weighted average."""
        t1 = make_task("t1", n_eval_docs=2)
        t2 = make_task("t2", n_eval_docs=4)

        group = Group(
            name="g",
            aggregate_metric_list=[AggMetricConfig(metric="acc", weight_by_size=True)],
        )
        group.add(t1)
        group.add(t2)

        agg = {"acc": mean}
        acc = {
            "t1": _make_acc(
                t1, {("acc", "none"): [1.0, 1.0]}, agg=agg
            ),  # mean=1.0, n=2
            "t2": _make_acc(
                t2, {("acc", "none"): [0.0, 0.0, 0.0, 1.0]}, agg=agg
            ),  # mean=0.25, n=4
        }
        result = _process_results(acc, groups={"g": group}, bootstrap_iters=0)

        # Weighted: (1.0*2 + 0.25*4) / (2+4) = 3.0/6 = 0.5
        assert result.metrics["g"]["acc,none"] == pytest.approx(0.5)
        assert result.metrics["g"]["sample_len"] == 6

    def test_two_tasks_unweighted_group(self, make_task):
        """Two tasks — unweighted means simple average of task means."""
        t1 = make_task("t1", n_eval_docs=2)
        t2 = make_task("t2", n_eval_docs=4)

        group = Group(
            name="g",
            aggregate_metric_list=[AggMetricConfig(metric="acc", weight_by_size=False)],
        )
        group.add(t1)
        group.add(t2)

        agg = {"acc": mean}
        acc = {
            "t1": _make_acc(t1, {("acc", "none"): [1.0, 1.0]}, agg=agg),  # mean=1.0
            "t2": _make_acc(
                t2, {("acc", "none"): [0.0, 0.0, 0.0, 1.0]}, agg=agg
            ),  # mean=0.25
        }
        result = _process_results(acc, groups={"g": group}, bootstrap_iters=0)

        # Unweighted: (1.0 + 0.25) / 2 = 0.625
        assert result.metrics["g"]["acc,none"] == pytest.approx(0.625)

    def test_sample_len_is_total_not_per_filter(self, make_task):
        """Group sample_len is total across all leaf tasks, not filter-dependent."""
        t1 = make_task("t1", n_eval_docs=3)
        t2 = make_task("t2", n_eval_docs=2)

        group = Group(name="g", aggregate_metric_list=[AggMetricConfig(metric="acc")])
        group.add(t1)
        group.add(t2)

        agg = {"acc": mean}
        # t1 has both filters, t2 only has "none"
        acc = {
            "t1": _make_acc(
                t1,
                {
                    ("acc", "none"): [0.0, 1.0, 1.0],
                    ("acc", "prefix"): [0.5, 0.5, 0.5],
                },
                agg=agg,
            ),
            "t2": _make_acc(
                t2,
                {
                    ("acc", "none"): [1.0, 1.0],
                },
                agg=agg,
            ),
        }
        result = _process_results(acc, groups={"g": group}, bootstrap_iters=0)

        # sample_len = total across all leaf tasks (3 + 2 = 5)
        assert result.metrics["g"]["sample_len"] == 5

        # sample_count tracks per-metric contributing docs
        assert result.metrics["g"]["sample_count"]["acc,none"] == 5  # both tasks
        assert result.metrics["g"]["sample_count"]["acc,prefix"] == 3  # only t1

    def test_multiple_metrics_sample_count(self, make_task):
        """sample_count is correct when tasks report different metrics."""
        t1 = make_task("t1", n_eval_docs=3)
        t2 = make_task("t2", n_eval_docs=2)

        group = Group(
            name="g",
            aggregate_metric_list=[
                AggMetricConfig(metric="acc"),
                AggMetricConfig(metric="f1"),
            ],
        )
        group.add(t1)
        group.add(t2)

        acc = {
            "t1": _make_acc(
                t1,
                {
                    ("acc", "none"): [0.8, 0.9, 0.7],
                    ("f1", "none"): [0.6, 0.7, 0.8],
                },
                agg={"acc": mean, "f1": mean},
            ),
            "t2": _make_acc(
                t2,
                {
                    ("acc", "none"): [0.5, 0.5],
                },
                agg={"acc": mean},
            ),
        }
        result = _process_results(acc, groups={"g": group}, bootstrap_iters=0)

        assert result.metrics["g"]["sample_len"] == 5
        assert result.metrics["g"]["sample_count"]["acc,none"] == 5  # both tasks
        assert result.metrics["g"]["sample_count"]["f1,none"] == 3  # only t1


class TestNestedGroupPipeline:
    """Tests for nested group hierarchies (parent → child → tasks)."""

    def test_two_level_hierarchy(self, make_task):
        """Parent group aggregates child group which aggregates tasks."""
        t1 = make_task("t1", n_eval_docs=2)
        t2 = make_task("t2", n_eval_docs=2)

        child = Group(
            name="child",
            aggregate_metric_list=[AggMetricConfig(metric="acc")],
        )
        child.add(t1)
        child.add(t2)

        parent = Group(
            name="parent",
            aggregate_metric_list=[AggMetricConfig(metric="acc")],
        )
        parent.add(child)

        agg = {"acc": mean}
        acc = {
            "t1": _make_acc(t1, {("acc", "none"): [1.0, 1.0]}, agg=agg),  # mean=1.0
            "t2": _make_acc(t2, {("acc", "none"): [0.0, 0.0]}, agg=agg),  # mean=0.0
        }
        result = _process_results(
            acc, groups={"parent": parent, "child": child}, bootstrap_iters=0
        )

        # Child: weighted avg of t1 and t2 = (1.0*2 + 0.0*2)/4 = 0.5
        assert result.metrics["child"]["acc,none"] == pytest.approx(0.5)
        assert result.metrics["child"]["sample_len"] == 4

        # Parent aggregates from leaf tasks (t1, t2), same result
        assert result.metrics["parent"]["acc,none"] == pytest.approx(0.5)
        assert result.metrics["parent"]["sample_len"] == 4

    def test_parent_with_mixed_children(self, make_task):
        """Parent has both a subgroup and a direct task."""
        t1 = make_task("t1", n_eval_docs=2)
        t2 = make_task("t2", n_eval_docs=2)

        subgroup = Group(
            name="sub",
            aggregate_metric_list=[AggMetricConfig(metric="acc")],
        )
        subgroup.add(t1)

        parent = Group(
            name="parent",
            aggregate_metric_list=[AggMetricConfig(metric="acc")],
        )
        parent.add(subgroup)
        parent.add(t2)

        agg = {"acc": mean}
        acc = {
            "t1": _make_acc(t1, {("acc", "none"): [0.8, 0.6]}, agg=agg),  # mean=0.7
            "t2": _make_acc(t2, {("acc", "none"): [1.0, 1.0]}, agg=agg),  # mean=1.0
        }
        result = _process_results(
            acc, groups={"parent": parent, "sub": subgroup}, bootstrap_iters=0
        )

        # Subgroup: only t1 → 0.7
        assert result.metrics["sub"]["acc,none"] == pytest.approx(0.7)

        # Parent: leaf tasks are t1 + t2 → (0.7*2 + 1.0*2)/4 = 0.85
        assert result.metrics["parent"]["acc,none"] == pytest.approx(0.85)
        assert result.metrics["parent"]["sample_len"] == 4


class TestGroupStderrPipeline:
    """Tests that stderr flows correctly through the pipeline."""

    def test_group_stderr_aggregated(self, make_task):
        """Group produces pooled stderr from task stderrs."""
        t1 = make_task("t1", n_eval_docs=50)
        t2 = make_task("t2", n_eval_docs=50)

        group = Group(name="g", aggregate_metric_list=[AggMetricConfig(metric="acc")])
        group.add(t1)
        group.add(t2)

        agg = {"acc": mean}
        # Use enough samples to get meaningful stderrs
        acc = {
            "t1": _make_acc(t1, {("acc", "none"): [1.0, 0.0] * 25}, agg=agg),
            "t2": _make_acc(t2, {("acc", "none"): [1.0, 0.0] * 25}, agg=agg),
        }
        result = _process_results(acc, groups={"g": group}, bootstrap_iters=100)

        # Task stderrs should exist
        assert isinstance(result.metrics["t1"]["acc_stderr,none"], float)
        assert isinstance(result.metrics["t2"]["acc_stderr,none"], float)

        # Group stderr should be computed (not N/A)
        assert isinstance(result.metrics["g"]["acc_stderr,none"], float)
        assert result.metrics["g"]["acc_stderr,none"] > 0

    def test_group_stderr_na_when_task_has_single_sample(self, make_task):
        """If a task has only 1 sample, its stderr is N/A → group stderr is N/A."""
        t1 = make_task("t1", n_eval_docs=1)

        group = Group(name="g", aggregate_metric_list=[AggMetricConfig(metric="acc")])
        group.add(t1)

        acc = {"t1": _make_acc(t1, {("acc", "none"): [1.0]}, agg={"acc": mean})}
        result = _process_results(acc, groups={"g": group}, bootstrap_iters=100)

        assert result.metrics["t1"]["acc_stderr,none"] == "N/A"
        assert result.metrics["g"]["acc_stderr,none"] == "N/A"


class TestGroupAggregationWarnings:
    """Tests for warning logs during group aggregation."""

    def test_warns_when_metric_missing_in_some_tasks(self, make_task, caplog):
        """Log warning when a metric exists in some tasks but not others."""
        t1 = make_task("t1", n_eval_docs=3)
        t2 = make_task("t2", n_eval_docs=2)

        group = Group(name="g", aggregate_metric_list=[AggMetricConfig(metric="acc")])
        group.add(t1)
        group.add(t2)

        agg = {"acc": mean}
        # t1 has both filters, t2 only has "none" → warning for "prefix"
        acc = {
            "t1": _make_acc(
                t1,
                {
                    ("acc", "none"): [0.8, 0.9, 0.7],
                    ("acc", "prefix"): [0.5, 0.5, 0.5],
                },
                agg=agg,
            ),
            "t2": _make_acc(
                t2,
                {
                    ("acc", "none"): [1.0, 1.0],
                },
                agg=agg,
            ),
        }
        with caplog.at_level(logging.WARNING):
            _process_results(acc, groups={"g": group}, bootstrap_iters=0)

        warning_messages = [
            r.message for r in caplog.records if r.levelname == "WARNING"
        ]
        # Should warn about "acc,prefix" missing in t2
        prefix_warnings = [m for m in warning_messages if "acc,prefix" in m]
        assert len(prefix_warnings) == 1
        assert "t2" in prefix_warnings[0]
        assert "1/2" in prefix_warnings[0]

    def test_warns_when_metric_missing_in_all_tasks(self, make_task, caplog):
        """Log warning when no tasks have the requested metric."""
        t1 = make_task("t1", n_eval_docs=2)

        group = Group(
            name="g",
            aggregate_metric_list=[
                AggMetricConfig(metric="nonexistent", filter_list=["none"]),
            ],
        )
        group.add(t1)

        acc = {"t1": _make_acc(t1, {("acc", "none"): [0.5, 0.5]}, agg={"acc": mean})}
        with caplog.at_level(logging.WARNING):
            _process_results(acc, groups={"g": group}, bootstrap_iters=0)

        warning_messages = [
            r.message for r in caplog.records if r.levelname == "WARNING"
        ]
        missing_warnings = [m for m in warning_messages if "no values found" in m]
        assert len(missing_warnings) == 1
        assert "nonexistent,none" in missing_warnings[0]

    def test_no_warning_when_all_tasks_have_metric(self, make_task, caplog):
        """No warning when every task has the metric."""
        t1 = make_task("t1", n_eval_docs=2)
        t2 = make_task("t2", n_eval_docs=2)

        group = Group(name="g", aggregate_metric_list=[AggMetricConfig(metric="acc")])
        group.add(t1)
        group.add(t2)

        agg = {"acc": mean}
        acc = {
            "t1": _make_acc(t1, {("acc", "none"): [0.8, 0.9]}, agg=agg),
            "t2": _make_acc(t2, {("acc", "none"): [0.7, 0.6]}, agg=agg),
        }
        with caplog.at_level(logging.WARNING):
            _process_results(acc, groups={"g": group}, bootstrap_iters=0)

        group_warnings = [
            r.message
            for r in caplog.records
            if r.levelname == "WARNING"
            and "g" in r.message
            and "missing" in r.message.lower()
        ]
        assert len(group_warnings) == 0


# ---------------------------------------------------------------------------
# TestProcessResultsBugFix
# ---------------------------------------------------------------------------


def _make_instance(doc_id: int, *, doc: dict, target: str, filter_key: str, resps):
    """Build an Instance with filtered_resps pre-populated."""
    from lm_eval.api.instance import Instance

    inst = Instance(
        request_type="loglikelihood",
        doc=doc,
        arguments=("ctx", "cont"),
        task_name="test_task",
        doc_id=doc_id,
        target=target,
    )
    inst.filtered_resps[filter_key] = resps
    return inst


class TestProcessResultsBugFix:
    """Regression test: legacy process_results path must populate references.

    Previously, the legacy path failed to populate references on ScoredDoc,
    causing ``reduce()`` to crash on ``zip(..., strict=True)``.
    ``_try_process_results`` now builds ScoredDoc objects with references
    from ``Instance.target``.
    """

    def test_scored_docs_have_references(self, make_task):
        """_try_process_results populates ScoredDoc.reference from Instance.target."""
        # process_results returns {"acc": score} for each doc
        scores_by_doc = {0: 1.0, 1: 0.0, 2: 1.0}
        task = make_task("t_legacy")
        task.process_results = lambda doc, results: {"acc": results[0]}

        instances: dict[int, list] = {}
        for doc_id in range(3):
            inst = _make_instance(
                doc_id,
                doc={"text": f"doc{doc_id}"},
                target=f"target_{doc_id}",
                filter_key="none",
                resps=[scores_by_doc[doc_id]],
            )
            instances[doc_id] = [inst]

        scored_docs = task._try_process_results(instances, filter_key="none")

        assert scored_docs is not None
        assert len(scored_docs) == 3
        for i, sd in scored_docs.items():
            assert isinstance(sd, ScoredDoc)
            assert sd.reference == f"target_{i}"
            assert "acc" in sd.scores
            assert sd.scores["acc"] == [scores_by_doc[i]]

    def test_path_reduce_succeeds(self, make_task):
        """Ensure reduce() works on ScoredDoc from the legacy path (no crash)."""
        from lm_eval.api.filter import FilterEnsemble
        from lm_eval.api.metrics import Metric

        task = make_task("t_legacy")
        task.process_results = lambda doc, results: {"acc": results[0]}

        instances: dict[int, list] = {}
        for doc_id, score in enumerate([1.0, 0.0]):
            inst = _make_instance(
                doc_id,
                doc={},
                target=f"ref_{doc_id}",
                filter_key="none",
                resps=[score],
            )
            instances[doc_id] = [inst]

        scored_docs = task._try_process_results(instances, filter_key="none")
        assert scored_docs is not None

        # Build a scorer and feed it the scored_docs — set_results should not crash
        noop_filter = FilterEnsemble("none", [("identity", None)])
        scorer = Scorer(
            name="none",
            filter=noop_filter,
            metrics=[
                Metric(
                    name="acc",
                    fn=lambda *a, **kw: 0,
                    aggregation=mean,
                    higher_is_better=True,
                )
            ],
        )

        scorer.set_results(scored_docs)
        reduced_values = [
            rd["acc"] for rd in scorer.reduced_docs.values() if "acc" in rd
        ]
        assert len(reduced_values) == 2
