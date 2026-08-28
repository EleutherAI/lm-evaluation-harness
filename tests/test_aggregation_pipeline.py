# type:ignore[invalid-assignment]
"""
End-to-end tests for the aggregation pipeline:
  raw per-sample values → task aggregation → group aggregation

Tests the full path through:
  _compute_task_aggregations → _collect_results → aggregate_groups
  (wrapped by _process_results)
"""

import logging
from typing import Any

import pytest

from lm_eval.api.group import AggMetricConfig, Group
from lm_eval.api.metrics import mean
from lm_eval.api.task import Task
from lm_eval.evaluator_utils import (
    ResultAcc,
    _process_results,
)
from lm_eval.result_schema import _TaskMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _m(d: dict[str, Any]) -> _TaskMetrics:
    """Cast a plain dict to _TaskMetrics for tests."""
    return d  # type: ignore[return-value]


class MockTask(Task):
    """Lightweight mock satisfying the Task ABC."""

    VERSION = 1

    def __init__(
        self,
        task_name: str,
        agg: dict | None = None,
        hib: dict | None = None,
        n_eval_docs: int = 100,
    ):
        self._task_name = task_name
        self._agg = agg or {}
        self._hib = hib or {}
        self._n_eval_docs = n_eval_docs

    @property
    def task_name(self):
        return self._task_name

    def dump_config(self) -> dict:
        return {"task_alias": self._task_name}

    def aggregation(self):
        return dict(self._agg)

    def higher_is_better(self):
        return dict(self._hib)

    @property
    def eval_docs(self):
        return [None] * self._n_eval_docs

    # ABC stubs
    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return [{}] * self._n_eval_docs

    def doc_to_text(self, doc):
        return ""

    def doc_to_target(self, doc):
        return ""

    def construct_requests(self, doc, ctx, **kwargs):
        return []

    def process_results(self, doc, results):
        return {}


def _make_acc(
    task: MockTask,
    raw_metrics: dict[tuple[str, str], list],
) -> ResultAcc:
    return {
        "task": task,
        "raw_metrics": raw_metrics,
        "logged_samples": [],
    }


# ---------------------------------------------------------------------------
# Pipeline: task raw values → _collect_results → group aggregation
# ---------------------------------------------------------------------------


class TestTaskToGroupPipeline:
    """End-to-end: raw per-sample values flow through task agg into group agg."""

    def test_single_task_single_group(self):
        """One task, one group — group metric == task metric."""
        task = MockTask("t1", agg={"acc": mean}, n_eval_docs=4)
        group = Group(name="g", aggregate_metric_list=[AggMetricConfig(metric="acc")])
        group.add(task)

        acc = {"t1": _make_acc(task, {("acc", "none"): [1.0, 0.0, 1.0, 0.0]})}
        result = _process_results(acc, groups={"g": group}, bootstrap_iters=0)

        # Task metrics
        assert result.metrics["t1"]["acc,none"] == pytest.approx(0.5)
        assert result.metrics["t1"]["sample_len"] == 4

        # Group metrics should match (single task)
        assert result.metrics["g"]["acc,none"] == pytest.approx(0.5)
        assert result.metrics["g"]["sample_len"] == 4

    def test_two_tasks_weighted_group(self):
        """Two tasks with different sizes — group uses weighted average."""
        t1 = MockTask("t1", agg={"acc": mean}, n_eval_docs=2)
        t2 = MockTask("t2", agg={"acc": mean}, n_eval_docs=4)

        group = Group(
            name="g",
            aggregate_metric_list=[AggMetricConfig(metric="acc", weight_by_size=True)],
        )
        group.add(t1)
        group.add(t2)

        acc = {
            "t1": _make_acc(t1, {("acc", "none"): [1.0, 1.0]}),  # mean=1.0, n=2
            "t2": _make_acc(
                t2, {("acc", "none"): [0.0, 0.0, 0.0, 1.0]}
            ),  # mean=0.25, n=4
        }
        result = _process_results(acc, groups={"g": group}, bootstrap_iters=0)

        # Weighted: (1.0*2 + 0.25*4) / (2+4) = 3.0/6 = 0.5
        assert result.metrics["g"]["acc,none"] == pytest.approx(0.5)
        assert result.metrics["g"]["sample_len"] == 6

    def test_two_tasks_unweighted_group(self):
        """Two tasks — unweighted means simple average of task means."""
        t1 = MockTask("t1", agg={"acc": mean}, n_eval_docs=2)
        t2 = MockTask("t2", agg={"acc": mean}, n_eval_docs=4)

        group = Group(
            name="g",
            aggregate_metric_list=[AggMetricConfig(metric="acc", weight_by_size=False)],
        )
        group.add(t1)
        group.add(t2)

        acc = {
            "t1": _make_acc(t1, {("acc", "none"): [1.0, 1.0]}),  # mean=1.0
            "t2": _make_acc(t2, {("acc", "none"): [0.0, 0.0, 0.0, 1.0]}),  # mean=0.25
        }
        result = _process_results(acc, groups={"g": group}, bootstrap_iters=0)

        # Unweighted: (1.0 + 0.25) / 2 = 0.625
        assert result.metrics["g"]["acc,none"] == pytest.approx(0.625)

    def test_sample_len_is_total_not_per_filter(self):
        """Group sample_len is total across all leaf tasks, not filter-dependent."""
        t1 = MockTask("t1", agg={"acc": mean}, n_eval_docs=3)
        t2 = MockTask("t2", agg={"acc": mean}, n_eval_docs=2)

        group = Group(name="g", aggregate_metric_list=[AggMetricConfig(metric="acc")])
        group.add(t1)
        group.add(t2)

        # t1 has both filters, t2 only has "none"
        acc = {
            "t1": _make_acc(
                t1,
                {
                    ("acc", "none"): [0.0, 1.0, 1.0],
                    ("acc", "prefix"): [0.5, 0.5, 0.5],
                },
            ),
            "t2": _make_acc(
                t2,
                {
                    ("acc", "none"): [1.0, 1.0],
                },
            ),
        }
        result = _process_results(acc, groups={"g": group}, bootstrap_iters=0)

        # sample_len = total across all leaf tasks (3 + 2 = 5)
        assert result.metrics["g"]["sample_len"] == 5

        # sample_count tracks per-metric contributing docs
        assert result.metrics["g"]["sample_count"]["acc,none"] == 5  # both tasks
        assert result.metrics["g"]["sample_count"]["acc,prefix"] == 3  # only t1

    def test_multiple_metrics_sample_count(self):
        """sample_count is correct when tasks report different metrics."""
        t1 = MockTask("t1", agg={"acc": mean, "f1": mean}, n_eval_docs=3)
        t2 = MockTask("t2", agg={"acc": mean}, n_eval_docs=2)

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
            ),
            "t2": _make_acc(
                t2,
                {
                    ("acc", "none"): [0.5, 0.5],
                },
            ),
        }
        result = _process_results(acc, groups={"g": group}, bootstrap_iters=0)

        assert result.metrics["g"]["sample_len"] == 5
        assert result.metrics["g"]["sample_count"]["acc,none"] == 5  # both tasks
        assert result.metrics["g"]["sample_count"]["f1,none"] == 3  # only t1


class TestNestedGroupPipeline:
    """Tests for nested group hierarchies (parent → child → tasks)."""

    def test_two_level_hierarchy(self):
        """Parent group aggregates child group which aggregates tasks."""
        t1 = MockTask("t1", agg={"acc": mean}, n_eval_docs=2)
        t2 = MockTask("t2", agg={"acc": mean}, n_eval_docs=2)

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

        acc = {
            "t1": _make_acc(t1, {("acc", "none"): [1.0, 1.0]}),  # mean=1.0
            "t2": _make_acc(t2, {("acc", "none"): [0.0, 0.0]}),  # mean=0.0
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

    def test_parent_with_mixed_children(self):
        """Parent has both a subgroup and a direct task."""
        t1 = MockTask("t1", agg={"acc": mean}, n_eval_docs=2)
        t2 = MockTask("t2", agg={"acc": mean}, n_eval_docs=2)

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

        acc = {
            "t1": _make_acc(t1, {("acc", "none"): [0.8, 0.6]}),  # mean=0.7
            "t2": _make_acc(t2, {("acc", "none"): [1.0, 1.0]}),  # mean=1.0
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

    def test_group_stderr_aggregated(self):
        """Group produces pooled stderr from task stderrs."""
        t1 = MockTask("t1", agg={"acc": mean}, n_eval_docs=50)
        t2 = MockTask("t2", agg={"acc": mean}, n_eval_docs=50)

        group = Group(name="g", aggregate_metric_list=[AggMetricConfig(metric="acc")])
        group.add(t1)
        group.add(t2)

        # Use enough samples to get meaningful stderrs
        acc = {
            "t1": _make_acc(t1, {("acc", "none"): [1.0, 0.0] * 25}),
            "t2": _make_acc(t2, {("acc", "none"): [1.0, 0.0] * 25}),
        }
        result = _process_results(acc, groups={"g": group}, bootstrap_iters=100)

        # Task stderrs should exist
        assert isinstance(result.metrics["t1"]["acc_stderr,none"], float)
        assert isinstance(result.metrics["t2"]["acc_stderr,none"], float)

        # Group stderr should be computed (not N/A)
        assert isinstance(result.metrics["g"]["acc_stderr,none"], float)
        assert result.metrics["g"]["acc_stderr,none"] > 0

    def test_group_stderr_na_when_task_has_single_sample(self):
        """If a task has only 1 sample, its stderr is N/A → group stderr is N/A."""
        t1 = MockTask("t1", agg={"acc": mean}, n_eval_docs=1)

        group = Group(name="g", aggregate_metric_list=[AggMetricConfig(metric="acc")])
        group.add(t1)

        acc = {"t1": _make_acc(t1, {("acc", "none"): [1.0]})}
        result = _process_results(acc, groups={"g": group}, bootstrap_iters=100)

        assert result.metrics["t1"]["acc_stderr,none"] == "N/A"
        assert result.metrics["g"]["acc_stderr,none"] == "N/A"


class TestGroupAggregationWarnings:
    """Tests for warning logs during group aggregation."""

    def test_warns_when_metric_missing_in_some_tasks(self, caplog):
        """Log warning when a metric exists in some tasks but not others."""
        t1 = MockTask("t1", agg={"acc": mean}, n_eval_docs=3)
        t2 = MockTask("t2", agg={"acc": mean}, n_eval_docs=2)

        group = Group(name="g", aggregate_metric_list=[AggMetricConfig(metric="acc")])
        group.add(t1)
        group.add(t2)

        # t1 has both filters, t2 only has "none" → warning for "prefix"
        acc = {
            "t1": _make_acc(
                t1,
                {
                    ("acc", "none"): [0.8, 0.9, 0.7],
                    ("acc", "prefix"): [0.5, 0.5, 0.5],
                },
            ),
            "t2": _make_acc(
                t2,
                {
                    ("acc", "none"): [1.0, 1.0],
                },
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

    def test_warns_when_metric_missing_in_all_tasks(self, caplog):
        """Log warning when no tasks have the requested metric."""
        t1 = MockTask("t1", agg={"acc": mean}, n_eval_docs=2)

        group = Group(
            name="g",
            aggregate_metric_list=[
                AggMetricConfig(metric="nonexistent", filter_list=["none"]),
            ],
        )
        group.add(t1)

        acc = {"t1": _make_acc(t1, {("acc", "none"): [0.5, 0.5]})}
        with caplog.at_level(logging.WARNING):
            _process_results(acc, groups={"g": group}, bootstrap_iters=0)

        warning_messages = [
            r.message for r in caplog.records if r.levelname == "WARNING"
        ]
        missing_warnings = [m for m in warning_messages if "no values found" in m]
        assert len(missing_warnings) == 1
        assert "nonexistent,none" in missing_warnings[0]

    def test_no_warning_when_all_tasks_have_metric(self, caplog):
        """No warning when every task has the metric."""
        t1 = MockTask("t1", agg={"acc": mean}, n_eval_docs=2)
        t2 = MockTask("t2", agg={"acc": mean}, n_eval_docs=2)

        group = Group(name="g", aggregate_metric_list=[AggMetricConfig(metric="acc")])
        group.add(t1)
        group.add(t2)

        acc = {
            "t1": _make_acc(t1, {("acc", "none"): [0.8, 0.9]}),
            "t2": _make_acc(t2, {("acc", "none"): [0.7, 0.6]}),
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
