"""
Tests for evaluator_utils.py — utility functions used by the evaluation pipeline.
"""

import logging

import pytest

from lm_eval.api.group import AggMetricConfig, Group
from lm_eval.api.metrics import mean
from lm_eval.api.task import Task
from lm_eval.evaluator_utils import (
    EvalResults,
    ResultAcc,
    _collect_groups_bottom_up,
    aggregate_groups,
    collect_results,
    format_results,
    get_results_data,
    get_root_groups,
    get_sample_size,
    process_results,
    propagate_higher_is_better,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
        r = EvalResults()
        assert r.metrics == {}
        assert r.configs == {}
        assert r.versions == {}
        assert r.num_fewshot == {}
        assert r.higher_is_better == {}
        assert r.samples == {}
        assert r.groups == {}

    def test_fields_are_independent_instances(self):
        a = EvalResults()
        b = EvalResults()
        a.metrics["x"] = {"v": 1}
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

from lm_eval.evaluator_utils import compute_task_aggregations


class TestComputeTaskAggregations:
    def _task(self, agg=None):
        return MockEvalTask("t", agg=agg or {"acc": mean})

    def test_single_metric_mean_aggregation(self):
        task = self._task()
        raw = {("acc", "none"): [0.0, 1.0, 1.0, 0.0]}
        metrics, count = compute_task_aggregations(task, raw, bootstrap_iters=0)
        assert metrics["acc,none"] == pytest.approx(0.5)
        assert count == 4

    def test_stderr_with_bootstrap_iters_zero(self):
        task = self._task()
        raw = {("acc", "none"): [0.0, 1.0]}
        metrics, _ = compute_task_aggregations(task, raw, bootstrap_iters=0)
        assert metrics["acc_stderr,none"] == "N/A"

    def test_stderr_with_bootstrap_iters_none(self):
        task = self._task()
        raw = {("acc", "none"): [0.0, 1.0]}
        metrics, _ = compute_task_aggregations(task, raw, bootstrap_iters=None)
        assert metrics["acc_stderr,none"] == "N/A"

    def test_stderr_with_positive_bootstrap_iters(self):
        task = self._task()
        raw = {("acc", "none"): [0.0, 1.0, 1.0, 0.0, 1.0]}
        metrics, _ = compute_task_aggregations(task, raw, bootstrap_iters=100)
        assert isinstance(metrics["acc_stderr,none"], float)

    def test_stderr_na_for_single_sample(self):
        task = self._task()
        raw = {("acc", "none"): [1.0]}
        metrics, _ = compute_task_aggregations(task, raw, bootstrap_iters=100)
        # len(items) <= 1 → "N/A"
        assert metrics["acc_stderr,none"] == "N/A"

    def test_fallback_to_mean_for_unknown_metric(self):
        # Task has no aggregation for "custom_metric"
        task = MockEvalTask("t", agg={})
        raw = {("custom_metric", "none"): [2.0, 4.0]}
        metrics, _ = compute_task_aggregations(task, raw, bootstrap_iters=0)
        assert metrics["custom_metric,none"] == pytest.approx(3.0)

    def test_multiple_metrics_and_filters(self):
        task = MockEvalTask("t", agg={"acc": mean, "f1": mean})
        raw = {
            ("acc", "none"): [1.0, 0.0],
            ("f1", "exact"): [0.8, 0.6],
        }
        metrics, _ = compute_task_aggregations(task, raw, bootstrap_iters=0)
        assert "acc,none" in metrics
        assert "f1,exact" in metrics
        assert metrics["acc,none"] == pytest.approx(0.5)
        assert metrics["f1,exact"] == pytest.approx(0.7)

    def test_sample_count_from_last_metric(self):
        task = MockEvalTask("t", agg={"a": mean, "b": mean})
        raw = {
            ("a", "none"): [1.0, 2.0],
            ("b", "none"): [1.0, 2.0, 3.0],
        }
        _, count = compute_task_aggregations(task, raw, bootstrap_iters=0)
        # sample_len is set by last iteration — dict is insertion‐ordered
        assert count == 3

    def test_bleu_metric_bootstrap_cap(self):
        task = MockEvalTask("t", agg={"bleu": mean})
        raw = {("bleu", "none"): [0.5, 0.6, 0.7]}
        # Should not raise; bootstrap_iters is capped to 100 internally
        metrics, _ = compute_task_aggregations(task, raw, bootstrap_iters=200)
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
        result = collect_results({"my_task": acc}, bootstrap_iters=0)
        assert "my_task" in result.metrics
        m = result.metrics["my_task"]
        assert "acc,none" in m
        assert m["alias"] == "My Task"
        assert m["samples"] == 4

    def test_alias_from_task_config(self):
        task, acc = self._simple_acc()
        result = collect_results({"my_task": acc}, bootstrap_iters=0)
        assert result.metrics["my_task"]["alias"] == "My Task"

    def test_alias_defaults_to_task_name(self):
        task = MockEvalTask("fallback_task", agg={"acc": mean})
        raw = {("acc", "none"): [1.0]}
        acc = make_result_acc(task, raw)
        result = collect_results({"fallback_task": acc}, bootstrap_iters=0)
        assert result.metrics["fallback_task"]["alias"] == "fallback_task"

    def test_configs_populated(self):
        task, acc = self._simple_acc()
        result = collect_results({"my_task": acc}, bootstrap_iters=0)
        assert result.configs["my_task"] == task.dump_config()

    def test_versions_populated(self):
        task, acc = self._simple_acc()
        result = collect_results({"my_task": acc}, bootstrap_iters=0)
        assert result.versions["my_task"] == 1

    def test_num_fewshot_populated(self):
        task, acc = self._simple_acc()
        result = collect_results({"my_task": acc}, bootstrap_iters=0)
        assert result.num_fewshot["my_task"] == 5

    def test_higher_is_better_populated(self):
        task, acc = self._simple_acc()
        result = collect_results({"my_task": acc}, bootstrap_iters=0)
        assert result.higher_is_better["my_task"] == {"acc": True}

    def test_samples_populated(self):
        task, acc = self._simple_acc()
        result = collect_results({"my_task": acc}, bootstrap_iters=0)
        assert result.samples["my_task"] == ["s1", "s2"]

    def test_groups_stored(self):
        task, acc = self._simple_acc()
        g = Group(name="grp")
        result = collect_results({"my_task": acc}, groups={"grp": g}, bootstrap_iters=0)
        assert result.groups == {"grp": g}

    def test_groups_default_to_empty(self):
        task, acc = self._simple_acc()
        result = collect_results({"my_task": acc}, groups=None, bootstrap_iters=0)
        assert result.groups == {}

    def test_multiple_tasks(self):
        t1 = MockEvalTask("t1", agg={"acc": mean}, hib={"acc": True})
        t2 = MockEvalTask("t2", agg={"acc": mean}, hib={"acc": False})
        accs = {
            "t1": make_result_acc(t1, {("acc", "none"): [1.0]}),
            "t2": make_result_acc(t2, {("acc", "none"): [0.0]}),
        }
        result = collect_results(accs, bootstrap_iters=0)
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
        roots = get_root_groups({"root": g})
        assert roots == [g]

    def test_root_excludes_children(self):
        child = Group(name="child")
        parent = Group(name="parent")
        parent.add(child)
        roots = get_root_groups({"parent": parent, "child": child})
        assert roots == [parent]

    def test_multiple_independent_roots(self):
        a = Group(name="a")
        b = Group(name="b")
        roots = get_root_groups({"a": a, "b": b})
        names = {g.name for g in roots}
        assert names == {"a", "b"}

    def test_empty_groups(self):
        assert get_root_groups({}) == []

    def test_deep_hierarchy(self):
        grandchild = Group(name="gc")
        child = Group(name="c")
        child.add(grandchild)
        grandparent = Group(name="gp")
        grandparent.add(child)
        roots = get_root_groups({"gp": grandparent, "c": child, "gc": grandchild})
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
        roots = get_root_groups(all_groups)
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
            aggregation=[AggMetricConfig(metric="acc")],
        )
        g.add(task)

        results = EvalResults()
        results.metrics["t1"] = {
            "alias": "T1",
            "samples": 100,
            "acc,none": 0.8,
            "acc_stderr,none": 0.02,
        }
        results.groups = {"grp": g}

        aggregate_groups(results)
        assert "grp" in results.metrics
        assert "acc,none" in results.metrics["grp"]

    def test_no_groups_noop(self):
        results = EvalResults()
        results.metrics["t"] = {"acc,none": 0.5}
        results.groups = {}
        aggregate_groups(results)
        assert "t" in results.metrics
        assert len(results.metrics) == 1

    def test_bottom_up_aggregation(self):
        """Child group aggregates before parent group."""
        task = MockEvalTask("leaf")
        child = Group(
            name="child",
            aggregation=[AggMetricConfig(metric="acc")],
        )
        child.add(task)

        parent = Group(
            name="parent",
            aggregation=[AggMetricConfig(metric="acc")],
        )
        parent.add(child)

        results = EvalResults()
        results.metrics["leaf"] = {
            "alias": "Leaf",
            "samples": 50,
            "acc,none": 0.9,
            "acc_stderr,none": 0.01,
        }
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
        result = process_results(self._basic_acc(), bootstrap_iters=0)
        assert isinstance(result, EvalResults)

    def test_with_groups(self):
        task = MockEvalTask("t", agg={"acc": mean})
        g = Group(name="g", aggregation=[AggMetricConfig(metric="acc")])
        g.add(task)
        acc = {"t": make_result_acc(task, {("acc", "none"): [0.5, 0.5]})}
        result = process_results(acc, groups={"g": g}, bootstrap_iters=0)
        assert "g" in result.metrics

    def test_without_groups(self):
        result = process_results(self._basic_acc(), groups=None, bootstrap_iters=0)
        assert result.groups == {}
        assert "t" in result.metrics


# ---------------------------------------------------------------------------
# TestFormatResults
# ---------------------------------------------------------------------------


class TestFormatResults:
    def test_standalone_task_formatting(self):
        results = EvalResults()
        results.metrics["standalone"] = {"alias": "standalone", "acc,none": 0.9}
        task_res, group_res = format_results(results)
        assert "standalone" in task_res
        assert not task_res["standalone"]["alias"].startswith(" ")

    def test_group_with_children_indentation(self):
        task = MockEvalTask("child_task")
        g = Group(name="grp", aggregation=[AggMetricConfig(metric="acc")])
        g.add(task)

        results = EvalResults()
        results.metrics["grp"] = {"alias": "grp", "acc,none": 0.85}
        results.metrics["child_task"] = {"alias": "child_task", "acc,none": 0.85}
        results.groups = {"grp": g}

        task_res, _ = format_results(results)
        assert task_res["child_task"]["alias"].startswith(" - ")

    def test_show_groups_true(self):
        task = MockEvalTask("t")
        g = Group(name="grp", aggregation=[AggMetricConfig(metric="acc")])
        g.add(task)

        results = EvalResults()
        results.metrics["grp"] = {"alias": "grp", "acc,none": 0.8}
        results.metrics["t"] = {"alias": "t", "acc,none": 0.8}
        results.groups = {"grp": g}

        task_res, group_res = format_results(results, show_groups=True)
        assert "grp" in task_res
        assert "grp" in group_res

    def test_show_groups_false(self):
        task = MockEvalTask("t")
        g = Group(name="grp", aggregation=[AggMetricConfig(metric="acc")])
        g.add(task)

        results = EvalResults()
        results.metrics["grp"] = {"alias": "grp", "acc,none": 0.8}
        results.metrics["t"] = {"alias": "t", "acc,none": 0.8}
        results.groups = {"grp": g}

        task_res, group_res = format_results(results, show_groups=False)
        assert "grp" in task_res
        assert "grp" not in group_res

    def test_group_without_aggregation_not_in_group_results(self):
        task = MockEvalTask("t")
        g = Group(name="grp")  # no aggregation
        g.add(task)

        results = EvalResults()
        results.metrics["grp"] = {"alias": "grp"}
        results.metrics["t"] = {"alias": "t", "acc,none": 0.8}
        results.groups = {"grp": g}

        _, group_res = format_results(results, show_groups=True)
        assert "grp" not in group_res

    def test_alias_used_in_formatting(self):
        results = EvalResults()
        results.metrics["task_x"] = {"alias": "Pretty Name", "acc,none": 0.5}
        task_res, _ = format_results(results)
        assert task_res["task_x"]["alias"] == "Pretty Name"

    def test_samples_removed_from_formatted_output(self):
        results = EvalResults()
        results.metrics["t"] = {"alias": "t", "samples": 100, "acc,none": 0.5}
        task_res, _ = format_results(results)
        assert "samples" not in task_res["t"]

    def test_multiple_roots_sorted(self):
        g_b = Group(name="b_group")
        g_a = Group(name="a_group")

        results = EvalResults()
        results.metrics["b_group"] = {"alias": "b_group"}
        results.metrics["a_group"] = {"alias": "a_group"}
        results.groups = {"b_group": g_b, "a_group": g_a}

        task_res, _ = format_results(results)
        keys = list(task_res.keys())
        assert keys.index("a_group") < keys.index("b_group")

    def test_standalone_tasks_sorted(self):
        results = EvalResults()
        results.metrics["z_task"] = {"alias": "z_task", "acc,none": 0.1}
        results.metrics["a_task"] = {"alias": "a_task", "acc,none": 0.2}
        task_res, _ = format_results(results)
        keys = list(task_res.keys())
        assert keys.index("a_task") < keys.index("z_task")


# ---------------------------------------------------------------------------
# TestGetResultsData
# ---------------------------------------------------------------------------


class TestGetResultsData:
    def test_strips_samples(self):
        results = EvalResults()
        results.metrics["t"] = {"alias": "t", "samples": 100, "acc,none": 0.9}
        task_res, _ = get_results_data(results)
        assert "samples" not in task_res["t"]

    def test_alias_not_indented(self):
        task = MockEvalTask("child_task")
        g = Group(name="grp", aggregation=[AggMetricConfig(metric="acc")])
        g.add(task)

        results = EvalResults()
        results.metrics["grp"] = {"alias": "grp", "samples": 50, "acc,none": 0.85}
        results.metrics["child_task"] = {
            "alias": "child_task",
            "samples": 50,
            "acc,none": 0.85,
        }
        results.groups = {"grp": g}

        task_res, group_res = get_results_data(results)
        # Aliases should be plain strings, no indentation
        assert task_res["grp"]["alias"] == "grp"
        assert task_res["child_task"]["alias"] == "child_task"
        assert group_res["grp"]["alias"] == "grp"

    def test_group_with_aggregation_in_group_results(self):
        task = MockEvalTask("t")
        g = Group(name="grp", aggregation=[AggMetricConfig(metric="acc")])
        g.add(task)

        results = EvalResults()
        results.metrics["grp"] = {"alias": "grp", "acc,none": 0.8}
        results.metrics["t"] = {"alias": "t", "acc,none": 0.8}
        results.groups = {"grp": g}

        task_res, group_res = get_results_data(results)
        assert "grp" in task_res
        assert "grp" in group_res

    def test_group_without_aggregation_not_in_group_results(self):
        task = MockEvalTask("t")
        g = Group(name="grp")  # no aggregation
        g.add(task)

        results = EvalResults()
        results.metrics["grp"] = {"alias": "grp"}
        results.metrics["t"] = {"alias": "t", "acc,none": 0.8}
        results.groups = {"grp": g}

        _, group_res = get_results_data(results)
        assert "grp" not in group_res

    def test_task_only_in_task_results(self):
        results = EvalResults()
        results.metrics["standalone"] = {"alias": "standalone", "acc,none": 0.9}
        task_res, group_res = get_results_data(results)
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
        propagate_higher_is_better([g], hib)
        assert hib["grp"] == {"acc": True}

    def test_conflicting_values_set_to_none(self):
        g = Group(name="grp")
        t1 = MockEvalTask("t1")
        t2 = MockEvalTask("t2")
        g.add(t1)
        g.add(t2)
        hib = {"t1": {"acc": True}, "t2": {"acc": False}}
        propagate_higher_is_better([g], hib)
        assert hib["grp"]["acc"] is None

    def test_conflicting_values_log_warning(self, caplog):
        g = Group(name="grp")
        t1 = MockEvalTask("t1")
        t2 = MockEvalTask("t2")
        g.add(t1)
        g.add(t2)
        hib = {"t1": {"acc": True}, "t2": {"acc": False}}
        with caplog.at_level(logging.WARNING):
            propagate_higher_is_better([g], hib)
        assert any("not consistent" in r.message for r in caplog.records)

    def test_no_children_in_higher_is_better(self):
        g = Group(name="grp")
        task = MockEvalTask("t")
        g.add(task)
        hib: dict = {}
        propagate_higher_is_better([g], hib)
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
        propagate_higher_is_better([g], hib)
        assert hib["grp"]["acc"] is True
        assert hib["grp"]["f1"] is None

    def test_empty_groups_list(self):
        hib: dict = {"t": {"acc": True}}
        propagate_higher_is_better([], hib)
        # Nothing changes
        assert hib == {"t": {"acc": True}}
