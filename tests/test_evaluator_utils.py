"""
Tests for evaluator_utils.py — utility functions used by the evaluation pipeline.
"""

import logging
from typing import Any

import pytest

from lm_eval.api.group import AggMetricConfig, Group
from lm_eval.api.metrics import mean
from lm_eval.api.task import Task
from lm_eval.evaluator_utils import (
    EvalAcc,
    ResultAcc,
    _collect_groups_bottom_up,
    _collect_results,
    _get_root_groups,
    _process_results,
    _propagate_higher_is_better,
    aggregate_groups,
    get_sample_size,
)
from lm_eval.result_schema import _TaskMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _m(d: dict[str, Any]) -> _TaskMetrics:
    """Cast a plain dict to TaskMetrics for tests (dynamic metric keys)."""
    return d  # type: ignore[return-value]


class MockEvalTask(Task):
    """Lightweight mock that satisfies the Task ABC for evaluator_utils tests."""

    VERSION = 1

    def __init__(
        self,
        task_name: str,
        config_dict: dict | None = None,
        agg: dict | None = None,
        hib: dict | None = None,
        n_eval_docs: int = 100,
    ):
        self._task_name = task_name
        self._config_dict = config_dict or {}
        self._agg = agg or {}
        self._hib = hib or {}
        self._n_eval_docs = n_eval_docs

    # -- identity ----------------------------------------------------------
    @property
    def task_name(self):
        return self._task_name

    # -- config / metadata -------------------------------------------------
    def dump_config(self) -> dict:
        return dict(self._config_dict)

    def aggregation(self):
        return dict(self._agg)

    def higher_is_better(self):
        return dict(self._hib)

    # -- eval_docs (used by get_sample_size) -------------------------------
    @property
    def eval_docs(self):
        return [None] * self._n_eval_docs

    # -- ABC stubs (never called in these tests) ---------------------------
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


def make_result_acc(
    task: MockEvalTask,
    raw_metrics: dict[tuple[str, str], list],
    logged_samples: list | None = None,
) -> ResultAcc:
    """Build a ResultAcc dict for use with collect_results."""
    return {
        "task": task,
        "raw_metrics": raw_metrics,
        "logged_samples": logged_samples or [],
    }


# ---------------------------------------------------------------------------
# TestEvalResults
# ---------------------------------------------------------------------------


class TestEvalResults:
    def test_default_fields_are_empty(self):
        r = EvalAcc()
        assert r.metrics == {}
        assert r.configs == {}
        assert r.versions == {}
        assert r.num_fewshot == {}
        assert r.higher_is_better == {}
        assert r.samples == {}
        assert r.n_samples == {}
        assert r.groups == {}

    def test_fields_are_independent_instances(self):
        a = EvalAcc()
        b = EvalAcc()
        a.metrics["x"] = _m({"v": 1})
        assert "x" not in b.metrics


# ---------------------------------------------------------------------------
# TestGetSampleSize
# ---------------------------------------------------------------------------


class TestGetSampleSize:
    def _task(self, n: int = 100):
        return MockEvalTask("t", n_eval_docs=n)

    def test_limit_none_returns_none(self):
        assert get_sample_size(self._task(), limit=None) is None

    def test_limit_integer_returns_int(self):
        assert get_sample_size(self._task(), limit=50) == 50

    def test_limit_fractional_rounds_up(self):
        # 100 * 0.3 = 30
        assert get_sample_size(self._task(100), limit=0.3) == 30

    def test_limit_fractional_small(self):
        # 10 * 0.05 = 0.5 → ceil → 1
        assert get_sample_size(self._task(10), limit=0.05) == 1

    def test_limit_one_is_treated_as_integer(self):
        # 1 is not < 1.0 so it goes to int(limit) branch
        assert get_sample_size(self._task(), limit=1) == 1

    def test_limit_float_exactly_one_is_integer(self):
        # 1.0 is not < 1.0 so treated as integer
        assert get_sample_size(self._task(), limit=1.0) == 1


# ---------------------------------------------------------------------------
# TestComputeTaskAggregations
# ---------------------------------------------------------------------------

from lm_eval.evaluator_utils import _compute_task_aggregations


class TestComputeTaskAggregations:
    def _task(self, agg=None):
        return MockEvalTask("t", agg=agg or {"acc": mean})

    def test_single_metric_mean_aggregation(self):
        task = self._task()
        raw = {("acc", "none"): [0.0, 1.0, 1.0, 0.0]}
        metrics, count = _compute_task_aggregations(task, raw, bootstrap_iters=0)
        assert metrics["acc,none"] == pytest.approx(0.5)
        assert count == 4

    def test_stderr_with_bootstrap_iters_zero(self):
        task = self._task()
        raw = {("acc", "none"): [0.0, 1.0]}
        metrics, _ = _compute_task_aggregations(task, raw, bootstrap_iters=0)
        assert metrics["acc_stderr,none"] == "N/A"

    def test_stderr_with_bootstrap_iters_none(self):
        task = self._task()
        raw = {("acc", "none"): [0.0, 1.0]}
        metrics, _ = _compute_task_aggregations(task, raw, bootstrap_iters=None)
        assert metrics["acc_stderr,none"] == "N/A"

    def test_stderr_with_positive_bootstrap_iters(self):
        task = self._task()
        raw = {("acc", "none"): [0.0, 1.0, 1.0, 0.0, 1.0]}
        metrics, _ = _compute_task_aggregations(task, raw, bootstrap_iters=100)
        assert isinstance(metrics["acc_stderr,none"], float)

    def test_stderr_na_for_single_sample(self):
        task = self._task()
        raw = {("acc", "none"): [1.0]}
        metrics, _ = _compute_task_aggregations(task, raw, bootstrap_iters=100)
        # len(items) <= 1 → "N/A"
        assert metrics["acc_stderr,none"] == "N/A"

    def test_fallback_to_mean_for_unknown_metric(self):
        # Task has no aggregation for "custom_metric"
        task = MockEvalTask("t", agg={})
        raw = {("custom_metric", "none"): [2.0, 4.0]}
        metrics, _ = _compute_task_aggregations(task, raw, bootstrap_iters=0)
        assert metrics["custom_metric,none"] == pytest.approx(3.0)

    def test_multiple_metrics_and_filters(self):
        task = MockEvalTask("t", agg={"acc": mean, "f1": mean})
        raw = {
            ("acc", "none"): [1.0, 0.0],
            ("f1", "exact"): [0.8, 0.6],
        }
        metrics, _ = _compute_task_aggregations(task, raw, bootstrap_iters=0)
        assert "acc,none" in metrics
        assert "f1,exact" in metrics
        assert metrics["acc,none"] == pytest.approx(0.5)
        assert metrics["f1,exact"] == pytest.approx(0.7)

    def test_bleu_metric_bootstrap_cap(self):
        task = MockEvalTask("t", agg={"bleu": mean})
        raw = {("bleu", "none"): [0.5, 0.6, 0.7]}
        # Should not raise; bootstrap_iters is capped to 100 internally
        metrics, _ = _compute_task_aggregations(task, raw, bootstrap_iters=200)
        assert "bleu,none" in metrics


# ---------------------------------------------------------------------------
# TestCollectResults
# ---------------------------------------------------------------------------


class TestCollectResults:
    def _simple_acc(self):
        task = MockEvalTask(
            "my_task",
            config_dict={"task_alias": "My Task", "num_fewshot": 5},
            agg={"acc": mean},
            hib={"acc": True},
        )
        raw = {("acc", "none"): [1.0, 0.0, 1.0, 1.0]}
        return task, make_result_acc(task, raw, logged_samples=["s1", "s2"])

    def test_single_task_basic_collection(self):
        task, acc = self._simple_acc()
        result = _collect_results({"my_task": acc}, bootstrap_iters=0)
        assert "my_task" in result.metrics
        m = result.metrics["my_task"]
        assert "acc,none" in m
        assert m["alias"] == "My Task"
        assert m["sample_len"] == 4

    def test_alias_from_task_config(self):
        task, acc = self._simple_acc()
        result = _collect_results({"my_task": acc}, bootstrap_iters=0)
        assert result.metrics["my_task"]["alias"] == "My Task"

    def test_alias_defaults_to_task_name(self):
        task = MockEvalTask("fallback_task", agg={"acc": mean})
        raw = {("acc", "none"): [1.0]}
        acc = make_result_acc(task, raw)
        result = _collect_results({"fallback_task": acc}, bootstrap_iters=0)
        assert result.metrics["fallback_task"]["alias"] == "fallback_task"

    def test_configs_populated(self):
        task, acc = self._simple_acc()
        result = _collect_results({"my_task": acc}, bootstrap_iters=0)
        assert result.configs["my_task"] == task.dump_config()

    def test_versions_populated(self):
        task, acc = self._simple_acc()
        result = _collect_results({"my_task": acc}, bootstrap_iters=0)
        assert result.versions["my_task"] == 1

    def test_num_fewshot_populated(self):
        task, acc = self._simple_acc()
        result = _collect_results({"my_task": acc}, bootstrap_iters=0)
        assert result.num_fewshot["my_task"] == 5

    def test_higher_is_better_populated(self):
        task, acc = self._simple_acc()
        result = _collect_results({"my_task": acc}, bootstrap_iters=0)
        assert result.higher_is_better["my_task"] == {"acc": True}

    def test_samples_populated(self):
        task, acc = self._simple_acc()
        result = _collect_results({"my_task": acc}, bootstrap_iters=0)
        assert result.samples["my_task"] == ["s1", "s2"]

    def test_groups_stored(self):
        task, acc = self._simple_acc()
        g = Group(name="grp")
        result = _collect_results(
            {"my_task": acc}, groups={"grp": g}, bootstrap_iters=0
        )
        assert result.groups == {"grp": g}

    def test_groups_default_to_empty(self):
        task, acc = self._simple_acc()
        result = _collect_results({"my_task": acc}, groups=None, bootstrap_iters=0)
        assert result.groups == {}

    def test_multiple_tasks(self):
        t1 = MockEvalTask("t1", agg={"acc": mean}, hib={"acc": True})
        t2 = MockEvalTask("t2", agg={"acc": mean}, hib={"acc": False})
        accs = {
            "t1": make_result_acc(t1, {("acc", "none"): [1.0]}),
            "t2": make_result_acc(t2, {("acc", "none"): [0.0]}),
        }
        result = _collect_results(accs, bootstrap_iters=0)
        assert "t1" in result.metrics
        assert "t2" in result.metrics
        assert "t1" in result.configs
        assert "t2" in result.configs


# ---------------------------------------------------------------------------
# TestGetRootGroups
# ---------------------------------------------------------------------------


class TestGetRootGroups:
    def test_single_root_group(self):
        g = Group(name="root")
        roots = _get_root_groups({"root": g})
        assert roots == [g]

    def test_root_excludes_children(self):
        child = Group(name="child")
        parent = Group(name="parent")
        parent.add(child)
        roots = _get_root_groups({"parent": parent, "child": child})
        assert roots == [parent]

    def test_multiple_independent_roots(self):
        a = Group(name="a")
        b = Group(name="b")
        roots = _get_root_groups({"a": a, "b": b})
        names = {g.name for g in roots}
        assert names == {"a", "b"}

    def test_empty_groups(self):
        assert _get_root_groups({}) == []

    def test_deep_hierarchy(self):
        grandchild = Group(name="gc")
        child = Group(name="c")
        child.add(grandchild)
        grandparent = Group(name="gp")
        grandparent.add(child)
        roots = _get_root_groups({"gp": grandparent, "c": child, "gc": grandchild})
        assert roots == [grandparent]

    def test_deep_hierarchy_multiple_roots(self):
        # Two independent hierarchies — only the two grandparents are roots
        gc1 = Group(name="gc1")
        c1 = Group(name="c1")
        c1.add(gc1)
        gp1 = Group(name="gp1")
        gp1.add(c1)

        gc2 = Group(name="gc2")
        c2 = Group(name="c2")
        c2.add(gc2)
        gp2 = Group(name="gp2")
        gp2.add(c2)

        all_groups = {
            "gp1": gp1,
            "c1": c1,
            "gc1": gc1,
            "gp2": gp2,
            "c2": c2,
            "gc2": gc2,
        }
        roots = _get_root_groups(all_groups)
        names = {g.name for g in roots}
        assert names == {"gp1", "gp2"}


# ---------------------------------------------------------------------------
# TestCollectGroupsBottomUp
# ---------------------------------------------------------------------------


class TestCollectGroupsBottomUp:
    def test_single_group_no_children(self):
        g = Group(name="g")
        result = _collect_groups_bottom_up({"g": g})
        assert result == [g]

    def test_parent_child_order(self):
        child = Group(name="child")
        parent = Group(name="parent")
        parent.add(child)
        result = _collect_groups_bottom_up({"parent": parent, "child": child})
        names = [g.name for g in result]
        assert names.index("child") < names.index("parent")

    def test_deep_hierarchy_order(self):
        gc = Group(name="gc")
        c = Group(name="c")
        c.add(gc)
        gp = Group(name="gp")
        gp.add(c)
        result = _collect_groups_bottom_up({"gp": gp, "c": c, "gc": gc})
        names = [g.name for g in result]
        assert names.index("gc") < names.index("c") < names.index("gp")

    def test_no_duplicates(self):
        shared = Group(name="shared")
        p1 = Group(name="p1")
        p1.add(shared)
        p2 = Group(name="p2")
        p2.add(shared)
        # Both parents reference "shared" — it should appear only once
        result = _collect_groups_bottom_up({"p1": p1, "p2": p2, "shared": shared})
        names = [g.name for g in result]
        assert names.count("shared") == 1

    def test_empty_groups(self):
        assert _collect_groups_bottom_up({}) == []


# ---------------------------------------------------------------------------
# TestAggregateGroups
# ---------------------------------------------------------------------------


class TestAggregateGroups:
    def test_group_metrics_added_to_results(self):
        task = MockEvalTask("t1")
        g = Group(
            name="grp",
            aggregate_metric_list=[AggMetricConfig(metric="acc")],
        )
        g.add(task)

        results = EvalAcc()
        results.metrics["t1"] = _m(
            {
                "alias": "T1",
                "sample_len": 100,
                "acc,none": 0.8,
                "acc_stderr,none": 0.02,
            }
        )
        results.groups = {"grp": g}

        aggregate_groups(results)
        assert "grp" in results.metrics
        assert "acc,none" in results.metrics["grp"]

    def test_no_groups_noop(self):
        results = EvalAcc()
        results.metrics["t"] = _m({"acc,none": 0.5})
        results.groups = {}
        aggregate_groups(results)
        assert "t" in results.metrics
        assert len(results.metrics) == 1

    def test_bottom_up_aggregation(self):
        """Child group aggregates before parent group."""
        task = MockEvalTask("leaf")
        child = Group(
            name="child",
            aggregate_metric_list=[AggMetricConfig(metric="acc")],
        )
        child.add(task)

        parent = Group(
            name="parent",
            aggregate_metric_list=[AggMetricConfig(metric="acc")],
        )
        parent.add(child)

        results = EvalAcc()
        results.metrics["leaf"] = _m(
            {
                "alias": "Leaf",
                "sample_len": 50,
                "acc,none": 0.9,
                "acc_stderr,none": 0.01,
            }
        )
        results.groups = {"parent": parent, "child": child}

        aggregate_groups(results)
        # Both child and parent should have metrics
        assert "child" in results.metrics
        assert "parent" in results.metrics


# ---------------------------------------------------------------------------
# TestProcessResults
# ---------------------------------------------------------------------------


class TestProcessResults:
    def _basic_acc(self):
        task = MockEvalTask("t", agg={"acc": mean}, hib={"acc": True})
        return {"t": make_result_acc(task, {("acc", "none"): [1.0, 0.0]})}

    def test_returns_eval_results(self):
        result = _process_results(self._basic_acc(), bootstrap_iters=0)
        assert isinstance(result, EvalAcc)

    def test_with_groups(self):
        task = MockEvalTask("t", agg={"acc": mean})
        g = Group(name="g", aggregate_metric_list=[AggMetricConfig(metric="acc")])
        g.add(task)
        acc = {"t": make_result_acc(task, {("acc", "none"): [0.5, 0.5]})}
        result = _process_results(acc, groups={"g": g}, bootstrap_iters=0)
        assert "g" in result.metrics

    def test_without_groups(self):
        result = _process_results(self._basic_acc(), groups=None, bootstrap_iters=0)
        assert result.groups == {}
        assert "t" in result.metrics


# ---------------------------------------------------------------------------
# TestGetResultsData
# ---------------------------------------------------------------------------


class TestGetResultsData:
    def test_preserves_sample_len(self):
        results = EvalAcc()
        results.metrics["t"] = _m({"alias": "t", "sample_len": 100, "acc,none": 0.9})
        task_res, _ = results.collect()
        assert task_res["t"]["sample_len"] == 100

    def test_alias_not_indented(self):
        task = MockEvalTask("child_task")
        g = Group(name="grp", aggregate_metric_list=[AggMetricConfig(metric="acc")])
        g.add(task)

        results = EvalAcc()
        results.metrics["grp"] = _m(
            {"alias": "grp", "sample_len": 50, "acc,none": 0.85}
        )
        results.metrics["child_task"] = _m(
            {
                "alias": "child_task",
                "sample_len": 50,
                "acc,none": 0.85,
            }
        )
        results.groups = {"grp": g}

        task_res, group_res = results.collect()
        # Aliases should be plain strings, no indentation
        assert task_res["grp"]["alias"] == "grp"
        assert task_res["child_task"]["alias"] == "child_task"
        assert group_res["grp"]["alias"] == "grp"

    def test_group_with_aggregation_in_group_results(self):
        task = MockEvalTask("t")
        g = Group(name="grp", aggregate_metric_list=[AggMetricConfig(metric="acc")])
        g.add(task)

        results = EvalAcc()
        results.metrics["grp"] = _m({"alias": "grp", "acc,none": 0.8})
        results.metrics["t"] = _m({"alias": "t", "acc,none": 0.8})
        results.groups = {"grp": g}

        task_res, group_res = results.collect()
        assert "grp" in task_res
        assert "grp" in group_res

    def test_group_without_aggregation_not_in_group_results(self):
        task = MockEvalTask("t")
        g = Group(name="grp")  # no aggregation
        g.add(task)

        results = EvalAcc()
        results.metrics["grp"] = _m({"alias": "grp"})
        results.metrics["t"] = _m({"alias": "t", "acc,none": 0.8})
        results.groups = {"grp": g}

        _, group_res = results.collect()
        assert "grp" not in group_res

    def test_task_only_in_task_results(self):
        results = EvalAcc()
        results.metrics["standalone"] = _m({"alias": "standalone", "acc,none": 0.9})
        task_res, group_res = results.collect()
        assert "standalone" in task_res
        assert "standalone" not in group_res


# ---------------------------------------------------------------------------
# TestPropagateHigherIsBetter
# ---------------------------------------------------------------------------


class TestPropagateHigherIsBetter:
    def test_propagation_from_children(self):
        g = Group(name="grp")
        task = MockEvalTask("t")
        g.add(task)
        hib = {"t": {"acc": True}}
        _propagate_higher_is_better([g], hib)
        assert hib["grp"] == {"acc": True}

    def test_conflicting_values_set_to_none(self):
        g = Group(name="grp")
        t1 = MockEvalTask("t1")
        t2 = MockEvalTask("t2")
        g.add(t1)
        g.add(t2)
        hib = {"t1": {"acc": True}, "t2": {"acc": False}}
        _propagate_higher_is_better([g], hib)
        assert hib["grp"]["acc"] is None

    def test_conflicting_values_log_warning(self, caplog):
        g = Group(name="grp")
        t1 = MockEvalTask("t1")
        t2 = MockEvalTask("t2")
        g.add(t1)
        g.add(t2)
        hib = {"t1": {"acc": True}, "t2": {"acc": False}}
        with caplog.at_level(logging.WARNING):
            _propagate_higher_is_better([g], hib)
        assert any("not consistent" in r.message for r in caplog.records)

    def test_no_children_in_higher_is_better(self):
        g = Group(name="grp")
        task = MockEvalTask("t")
        g.add(task)
        hib: dict = {}
        _propagate_higher_is_better([g], hib)
        # No child data → group should not appear
        assert "grp" not in hib

    def test_multiple_metrics_mixed(self):
        g = Group(name="grp")
        t1 = MockEvalTask("t1")
        t2 = MockEvalTask("t2")
        g.add(t1)
        g.add(t2)
        hib = {
            "t1": {"acc": True, "f1": True},
            "t2": {"acc": True, "f1": False},
        }
        _propagate_higher_is_better([g], hib)
        assert hib["grp"]["acc"] is True
        assert hib["grp"]["f1"] is None

    def test_empty_groups_list(self):
        hib: dict = {"t": {"acc": True}}
        _propagate_higher_is_better([], hib)
        # Nothing changes
        assert hib == {"t": {"acc": True}}


# ---------------------------------------------------------------------------
# TestToEvalResults
# ---------------------------------------------------------------------------


class TestToEvalResults:
    """Tests for EvalAcc.to_eval_results()."""

    def _make_eval_acc(self, *, with_group: bool = False, has_aggregation: bool = True):
        """Build a minimal EvalAcc for testing to_eval_results()."""
        task = MockEvalTask(
            "t1",
            config_dict={"task_alias": "Task One", "num_fewshot": 3},
            agg={"acc": mean},
            hib={"acc": True},
            n_eval_docs=100,
        )
        acc_input = {"t1": make_result_acc(task, {("acc", "none"): [1.0, 0.0, 1.0]})}

        groups = {}
        if with_group:
            if has_aggregation:
                g = Group(
                    name="grp", aggregate_metric_list=[AggMetricConfig(metric="acc")]
                )
            else:
                g = Group(name="grp")
            g.add(task)
            groups = {"grp": g}

        result = _process_results(acc_input, groups=groups, bootstrap_iters=0)
        return result

    def test_output_has_required_keys(self):
        er = self._make_eval_acc()
        d = er._to_eval_results()
        for key in (
            "results",
            "group_subtasks",
            "configs",
            "versions",
            "n-shot",
            "higher_is_better",
            "n-samples",
        ):
            assert key in d, f"Missing key: {key}"

    def test_results_contain_task_metrics(self):
        er = self._make_eval_acc()
        d = er._to_eval_results()
        assert "t1" in d["results"]
        assert "acc,none" in d["results"]["t1"]

    def test_n_samples_effective_from_sample_len(self):
        """Effective comes from sample_len (number of raw metric values)."""
        er = self._make_eval_acc()
        d = er._to_eval_results()
        assert d["n-samples"]["t1"]["original"] == 100
        # 3 raw metric values → sample_len == 3
        assert d["n-samples"]["t1"]["effective"] == 3

    def test_groups_key_present_when_group_has_aggregation(self):
        er = self._make_eval_acc(with_group=True, has_aggregation=True)
        d = er._to_eval_results()
        assert "groups" in d
        assert "grp" in d["groups"]

    def test_groups_key_absent_when_no_group_has_aggregation(self):
        er = self._make_eval_acc(with_group=True, has_aggregation=False)
        d = er._to_eval_results()
        assert "groups" not in d

    def test_groups_key_absent_when_no_groups(self):
        er = self._make_eval_acc(with_group=False)
        d = er._to_eval_results()
        assert "groups" not in d

    def test_samples_included_when_provided(self):
        er = self._make_eval_acc()
        d = er._to_eval_results(samples={"t1": [{"doc": 1}]})
        assert "samples" in d
        assert d["samples"]["t1"] == [{"doc": 1}]

    def test_samples_absent_when_not_provided(self):
        er = self._make_eval_acc()
        d = er._to_eval_results()
        assert "samples" not in d

    def test_higher_is_better_propagated_to_groups(self):
        er = self._make_eval_acc(with_group=True, has_aggregation=True)
        d = er._to_eval_results()
        assert "grp" in d["higher_is_better"]
        assert d["higher_is_better"]["grp"]["acc"] is True

    def test_configs_sorted(self):
        er = self._make_eval_acc()
        d = er._to_eval_results()
        assert list(d["configs"].keys()) == sorted(d["configs"].keys())

    def test_versions_sorted(self):
        er = self._make_eval_acc()
        d = er._to_eval_results()
        assert list(d["versions"].keys()) == sorted(d["versions"].keys())


# ---------------------------------------------------------------------------
# TestCollectResultsNSamples
# ---------------------------------------------------------------------------


class TestCollectResultsNSamples:
    """Tests for n_samples population via sample_len in collect_results()."""

    def test_n_samples_effective_equals_sample_len(self):
        t1 = MockEvalTask("t1", agg={"acc": mean}, n_eval_docs=100)
        t2 = MockEvalTask("t2", agg={"acc": mean}, n_eval_docs=200)
        accs = {
            "t1": make_result_acc(t1, {("acc", "none"): [1.0, 0.0, 1.0]}),
            "t2": make_result_acc(t2, {("acc", "none"): [0.0]}),
        }
        result = _collect_results(accs, bootstrap_iters=0)
        assert result.n_samples["t1"] == {"original": 100, "effective": 3}
        assert result.n_samples["t2"] == {"original": 200, "effective": 1}

    def test_n_samples_original_from_eval_docs(self):
        task = MockEvalTask("t1", agg={"acc": mean}, n_eval_docs=42)
        accs = {"t1": make_result_acc(task, {("acc", "none"): [1.0, 0.5]})}
        result = _collect_results(accs, bootstrap_iters=0)
        assert result.n_samples["t1"]["original"] == 42
        assert result.n_samples["t1"]["effective"] == 2
