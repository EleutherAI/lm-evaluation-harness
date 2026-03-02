from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import pytest

from lm_eval.api.group import Group
from lm_eval.api.metrics import mean
from lm_eval.config.group import AggMetricConfig
from lm_eval.evaluator_utils import (
    _agg_and_collect,
    _aggregate_groups,
    _collect_groups_bottom_up,
    _compute_task_aggregations,
    _EvalAcc,
    _get_root_groups,
    _get_sample_size,
    _process_results,
    _propagate_higher_is_better_,
)
from lm_eval.filters import build_filter_ensemble
from lm_eval.scorers import MetricKey, ScoredDoc, Scorer


if TYPE_CHECKING:
    from lm_eval.api.filter import FilterEnsemble
    from lm_eval.evaluator_utils import _ResultAcc
    from lm_eval.result_schema import _TaskMetrics


def _noop_filter(name: str = "none") -> FilterEnsemble:
    """Build a no-op filter ensemble for tests."""
    return build_filter_ensemble(name, [("noop", None)])


@pytest.fixture
def noop_filter() -> FilterEnsemble:
    return _noop_filter()


def _m(d: dict[str, Any]) -> _TaskMetrics:
    """Cast a plain dict to TaskMetrics for tests (dynamic metric keys)."""
    return d  # type: ignore[return-value]


def _build_mock_scorers(
    agg: dict[str, Any],
    hib: dict[str, bool] | None = None,
) -> list[Scorer]:
    """Build minimal Scorer objects from an {metric_name: agg_fn} dict.

    Returns one Scorer named "none" containing all metrics.
    """
    from lm_eval.api.metrics import Metric

    hib = hib or {}
    metrics = []
    for metric_name, agg_fn in agg.items():
        metrics.append(
            Metric(
                name=metric_name,
                fn=lambda *a, **kw: 0,  # unused in aggregation tests
                aggregation=agg_fn,
                higher_is_better=hib.get(metric_name, True),
            )
        )
    return [Scorer(name="none", filter=_noop_filter(), metrics=metrics)]


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
    """Build Scorer objects that match the tuple-keyed raw_metrics.

    Groups by scorer name, populates _reduced_docs.
    """
    from lm_eval.api.metrics import Metric

    agg = agg or {}
    hib = hib or {}

    # Group by scorer name
    scorers_data: dict[str, dict[str, list]] = defaultdict(dict)
    for (metric_name, scorer_name), values in raw_metrics.items():
        scorers_data[scorer_name][metric_name] = values

    scorers = []
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
            filter=_noop_filter(scorer_name),
            metrics=metrics,
        )
        scorer._reduced_docs = _reduced_docs_from_flat(metrics_dict)
        scorers.append(scorer)
    return scorers


def make_result_acc(
    task,
    raw_metrics: dict[tuple[str, str], list],
    agg: dict[str, Any] | None = None,
    hib: dict[str, bool] | None = None,
    logged_samples: list | None = None,
) -> _ResultAcc:
    """Build a _ResultAcc dict for use with collect_results.

    Also populates the task's _scorers with _reduced_docs.
    """
    task._scorers = _build_multi_scorer_scorers(
        raw_metrics, agg=agg or {}, hib=hib or {}
    )
    return {
        "task": task,
        "logged_samples": logged_samples or [],
    }


# ---------------------------------------------------------------------------
# TestEvalResults
# ---------------------------------------------------------------------------


class TestEvalResults:
    def test_default_fields_are_empty(self):
        r = _EvalAcc()
        assert r.metrics == {}
        assert r.configs == {}
        assert r.versions == {}
        assert r.num_fewshot == {}
        assert r.higher_is_better == {}
        assert r.samples == {}
        assert r.n_samples == {}
        assert r.groups == {}

    def test_fields_are_independent_instances(self):
        a = _EvalAcc()
        b = _EvalAcc()
        a.metrics["x"] = _m({"v": 1})
        assert "x" not in b.metrics


# ---------------------------------------------------------------------------
# TestGetSampleSize
# ---------------------------------------------------------------------------


class TestGetSampleSize:
    def test_limit_none_returns_none(self, make_task):
        assert _get_sample_size(make_task("t", n_eval_docs=100), limit=None) is None

    def test_limit_integer_returns_int(self, make_task):
        assert _get_sample_size(make_task("t", n_eval_docs=100), limit=50) == 50

    def test_limit_fractional_rounds_up(self, make_task):
        # 100 * 0.3 = 30
        assert _get_sample_size(make_task("t", n_eval_docs=100), limit=0.3) == 30

    def test_limit_fractional_small(self, make_task):
        # 10 * 0.05 = 0.5 → ceil → 1
        assert _get_sample_size(make_task("t", n_eval_docs=10), limit=0.05) == 1

    def test_limit_one_is_treated_as_integer(self, make_task):
        # 1 is not < 1.0 so it goes to int(limit) branch
        assert _get_sample_size(make_task("t", n_eval_docs=100), limit=1) == 1

    def test_limit_float_exactly_one_is_integer(self, make_task):
        # 1.0 is not < 1.0 so treated as integer
        assert _get_sample_size(make_task("t", n_eval_docs=100), limit=1.0) == 1


# ---------------------------------------------------------------------------
# TestComputeTaskAggregations
# ---------------------------------------------------------------------------


class TestComputeTaskAggregations:
    def _task_with_data(self, make_task, raw_metrics, agg=None):
        """Create a task with scorers pre-populated from raw_metrics."""
        agg = agg or {"acc": mean}
        task = make_task("t")
        task._scorers = _build_multi_scorer_scorers(raw_metrics, agg=agg)
        return task

    def test_single_metric_mean_aggregation(self, make_task):
        raw = {("acc", "none"): [0.0, 1.0, 1.0, 0.0]}
        task = self._task_with_data(make_task, raw)
        metrics, count = _compute_task_aggregations(task, bootstrap_iters=0)
        assert metrics["acc,none"] == pytest.approx(0.5)
        assert count == 4

    def test_stderr_with_bootstrap_iters_zero(self, make_task):
        raw = {("acc", "none"): [0.0, 1.0]}
        task = self._task_with_data(make_task, raw)
        metrics, _ = _compute_task_aggregations(task, bootstrap_iters=0)
        assert metrics["acc_stderr,none"] == "N/A"

    def test_stderr_with_bootstrap_iters_none(self, make_task):
        raw = {("acc", "none"): [0.0, 1.0]}
        task = self._task_with_data(make_task, raw)
        metrics, _ = _compute_task_aggregations(task, bootstrap_iters=None)
        assert metrics["acc_stderr,none"] == "N/A"

    def test_stderr_with_positive_bootstrap_iters(self, make_task):
        raw = {("acc", "none"): [0.0, 1.0, 1.0, 0.0, 1.0]}
        task = self._task_with_data(make_task, raw)
        metrics, _ = _compute_task_aggregations(task, bootstrap_iters=100)
        assert isinstance(metrics["acc_stderr,none"], float)

    def test_stderr_na_for_single_sample(self, make_task):
        raw = {("acc", "none"): [1.0]}
        task = self._task_with_data(make_task, raw)
        metrics, _ = _compute_task_aggregations(task, bootstrap_iters=100)
        # len(items) <= 1 → "N/A"
        assert metrics["acc_stderr,none"] == "N/A"

    def test_fallback_to_mean_for_unknown_metric(self, make_task):
        # Task has no aggregation for "custom_metric"
        raw = {("custom_metric", "none"): [2.0, 4.0]}
        task = make_task("t")
        task._scorers = _build_multi_scorer_scorers(raw, agg={})
        metrics, _ = _compute_task_aggregations(task, bootstrap_iters=0)
        assert metrics["custom_metric,none"] == pytest.approx(3.0)

    def test_multiple_metrics_and_filters(self, make_task):
        agg = {"acc": mean, "f1": mean}
        raw = {
            ("acc", "none"): [1.0, 0.0],
            ("f1", "exact"): [0.8, 0.6],
        }
        task = make_task("t")
        task._scorers = _build_multi_scorer_scorers(raw, agg=agg)
        metrics, _ = _compute_task_aggregations(task, bootstrap_iters=0)
        assert "acc,none" in metrics
        assert "f1,exact" in metrics
        assert metrics["acc,none"] == pytest.approx(0.5)
        assert metrics["f1,exact"] == pytest.approx(0.7)

    def test_bleu_metric_bootstrap_cap(self, make_task):
        agg = {"bleu": mean}
        raw = {("bleu", "none"): [0.5, 0.6, 0.7]}
        task = self._task_with_data(make_task, raw, agg=agg)
        # Should not raise; bootstrap_iters is capped to 100 internally
        metrics, _ = _compute_task_aggregations(task, bootstrap_iters=200)
        assert "bleu,none" in metrics


# ---------------------------------------------------------------------------
# TestCollectResults
# ---------------------------------------------------------------------------


class TestCollectResults:
    def _simple_acc(self, make_task):
        task = make_task(
            "my_task",
            task_alias="My Task",
            num_fewshot=5,
            metadata={"version": 1},
            n_eval_docs=100,
        )
        raw = {("acc", "none"): [1.0, 0.0, 1.0, 1.0]}
        agg, hib = {"acc": mean}, {"acc": True}
        return task, make_result_acc(
            task, raw, agg=agg, hib=hib, logged_samples=["s1", "s2"]
        )

    def test_single_task_basic_collection(self, make_task):
        task, acc = self._simple_acc(make_task)
        result = _agg_and_collect({"my_task": acc}, bootstrap_iters=0)
        assert "my_task" in result.metrics
        m = result.metrics["my_task"]
        assert "acc,none" in m
        assert m["alias"] == "My Task"
        assert m["sample_len"] == 4

    def test_alias_from_task_config(self, make_task):
        task, acc = self._simple_acc(make_task)
        result = _agg_and_collect({"my_task": acc}, bootstrap_iters=0)
        assert result.metrics["my_task"]["alias"] == "My Task"

    def test_alias_defaults_to_task_name(self, make_task):
        task = make_task("fallback_task", n_eval_docs=10)
        raw = {("acc", "none"): [1.0]}
        acc = make_result_acc(task, raw, agg={"acc": mean})
        result = _agg_and_collect({"fallback_task": acc}, bootstrap_iters=0)
        assert result.metrics["fallback_task"]["alias"] == "fallback_task"

    def test_configs_populated(self, make_task):
        task, acc = self._simple_acc(make_task)
        result = _agg_and_collect({"my_task": acc}, bootstrap_iters=0)
        assert result.configs["my_task"] == task.dump_config()

    def test_versions_populated(self, make_task):
        task, acc = self._simple_acc(make_task)
        result = _agg_and_collect({"my_task": acc}, bootstrap_iters=0)
        assert result.versions["my_task"] == 1

    def test_num_fewshot_populated(self, make_task):
        task, acc = self._simple_acc(make_task)
        result = _agg_and_collect({"my_task": acc}, bootstrap_iters=0)
        assert result.num_fewshot["my_task"] == 5

    def test_higher_is_better_populated(self, make_task):
        task, acc = self._simple_acc(make_task)
        result = _agg_and_collect({"my_task": acc}, bootstrap_iters=0)
        assert result.higher_is_better["my_task"] == {"acc": True}

    def test_samples_populated(self, make_task):
        task, acc = self._simple_acc(make_task)
        result = _agg_and_collect({"my_task": acc}, bootstrap_iters=0)
        assert result.samples["my_task"] == ["s1", "s2"]

    def test_groups_stored(self, make_task):
        task, acc = self._simple_acc(make_task)
        g = Group(name="grp")
        result = _agg_and_collect(
            {"my_task": acc}, groups={"grp": g}, bootstrap_iters=0
        )
        assert result.groups == {"grp": g}

    def test_groups_default_to_empty(self, make_task):
        task, acc = self._simple_acc(make_task)
        result = _agg_and_collect({"my_task": acc}, groups=None, bootstrap_iters=0)
        assert result.groups == {}

    def test_multiple_tasks(self, make_task):
        t1 = make_task("t1", n_eval_docs=10)
        t2 = make_task("t2", n_eval_docs=10)
        accs = {
            "t1": make_result_acc(
                t1, {("acc", "none"): [1.0]}, agg={"acc": mean}, hib={"acc": True}
            ),
            "t2": make_result_acc(
                t2, {("acc", "none"): [0.0]}, agg={"acc": mean}, hib={"acc": False}
            ),
        }
        result = _agg_and_collect(accs, bootstrap_iters=0)
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
    def test_group_metrics_added_to_results(self, make_task):
        task = make_task("t1")
        g = Group(
            name="grp",
            aggregate_metric_list=[AggMetricConfig(metric="acc")],
        )
        g.add(task)

        results = _EvalAcc()
        results.metrics["t1"] = _m(
            {
                "alias": "T1",
                "sample_len": 100,
                "acc,none": 0.8,
                "acc_stderr,none": 0.02,
            }
        )
        results.groups = {"grp": g}

        _aggregate_groups(results)
        assert "grp" in results.metrics
        assert "acc,none" in results.metrics["grp"]

    def test_no_groups_noop(self):
        results = _EvalAcc()
        results.metrics["t"] = _m({"acc,none": 0.5})
        results.groups = {}
        _aggregate_groups(results)
        assert "t" in results.metrics
        assert len(results.metrics) == 1

    def test_bottom_up_aggregation(self, make_task):
        """Child group aggregates before parent group."""
        task = make_task("leaf")
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

        results = _EvalAcc()
        results.metrics["leaf"] = _m(
            {
                "alias": "Leaf",
                "sample_len": 50,
                "acc,none": 0.9,
                "acc_stderr,none": 0.01,
            }
        )
        results.groups = {"parent": parent, "child": child}

        _aggregate_groups(results)
        # Both child and parent should have metrics
        assert "child" in results.metrics
        assert "parent" in results.metrics


# ---------------------------------------------------------------------------
# TestProcessResults
# ---------------------------------------------------------------------------


class TestProcessResults:
    def _basic_acc(self, make_task):
        task = make_task("t", n_eval_docs=10)
        return {
            "t": make_result_acc(
                task,
                {("acc", "none"): [1.0, 0.0]},
                agg={"acc": mean},
                hib={"acc": True},
            )
        }

    def test_returns_eval_results(self, make_task):
        result = _process_results(self._basic_acc(make_task), bootstrap_iters=0)
        assert isinstance(result, _EvalAcc)

    def test_with_groups(self, make_task):
        task = make_task("t", n_eval_docs=10)
        g = Group(name="g", aggregate_metric_list=[AggMetricConfig(metric="acc")])
        g.add(task)
        acc = {
            "t": make_result_acc(task, {("acc", "none"): [0.5, 0.5]}, agg={"acc": mean})
        }
        result = _process_results(acc, groups={"g": g}, bootstrap_iters=0)
        assert "g" in result.metrics

    def test_without_groups(self, make_task):
        result = _process_results(
            self._basic_acc(make_task), groups=None, bootstrap_iters=0
        )
        assert result.groups == {}
        assert "t" in result.metrics


# ---------------------------------------------------------------------------
# TestGetResultsData
# ---------------------------------------------------------------------------


class TestGetResultsData:
    def test_preserves_sample_len(self):
        results = _EvalAcc()
        results.metrics["t"] = _m({"alias": "t", "sample_len": 100, "acc,none": 0.9})
        task_res, _ = results.collect()
        assert task_res["t"]["sample_len"] == 100

    def test_alias_not_indented(self, make_task):
        task = make_task("child_task")
        g = Group(name="grp", aggregate_metric_list=[AggMetricConfig(metric="acc")])
        g.add(task)

        results = _EvalAcc()
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

    def test_group_with_aggregation_in_group_results(self, make_task):
        task = make_task("t")
        g = Group(name="grp", aggregate_metric_list=[AggMetricConfig(metric="acc")])
        g.add(task)

        results = _EvalAcc()
        results.metrics["grp"] = _m({"alias": "grp", "acc,none": 0.8})
        results.metrics["t"] = _m({"alias": "t", "acc,none": 0.8})
        results.groups = {"grp": g}

        task_res, group_res = results.collect()
        assert "grp" in task_res
        assert "grp" in group_res

    def test_group_without_aggregation_not_in_group_results(self, make_task):
        task = make_task("t")
        g = Group(name="grp")  # no aggregation
        g.add(task)

        results = _EvalAcc()
        results.metrics["grp"] = _m({"alias": "grp"})
        results.metrics["t"] = _m({"alias": "t", "acc,none": 0.8})
        results.groups = {"grp": g}

        _, group_res = results.collect()
        assert "grp" not in group_res

    def test_task_only_in_task_results(self):
        results = _EvalAcc()
        results.metrics["standalone"] = _m({"alias": "standalone", "acc,none": 0.9})
        task_res, group_res = results.collect()
        assert "standalone" in task_res
        assert "standalone" not in group_res


# ---------------------------------------------------------------------------
# TestPropagateHigherIsBetter
# ---------------------------------------------------------------------------


class TestPropagateHigherIsBetter:
    def test_propagation_from_children(self, make_task):
        g = Group(name="grp")
        g.add(make_task("t"))
        hib = {"t": {"acc": True}}
        _propagate_higher_is_better_([g], hib)
        assert hib["grp"] == {"acc": True}

    def test_conflicting_values_set_to_none(self, make_task):
        g = Group(name="grp")
        g.add(make_task("t1"))
        g.add(make_task("t2"))
        hib = {"t1": {"acc": True}, "t2": {"acc": False}}
        _propagate_higher_is_better_([g], hib)
        assert hib["grp"]["acc"] is None

    def test_conflicting_values_log_warning(self, make_task, caplog):
        g = Group(name="grp")
        g.add(make_task("t1"))
        g.add(make_task("t2"))
        hib = {"t1": {"acc": True}, "t2": {"acc": False}}
        with caplog.at_level(logging.WARNING):
            _propagate_higher_is_better_([g], hib)
        assert any("not consistent" in r.message for r in caplog.records)

    def test_no_children_in_higher_is_better(self, make_task):
        g = Group(name="grp")
        g.add(make_task("t"))
        hib: dict = {}
        _propagate_higher_is_better_([g], hib)
        # No child data → group should not appear
        assert "grp" not in hib

    def test_multiple_metrics_mixed(self, make_task):
        g = Group(name="grp")
        g.add(make_task("t1"))
        g.add(make_task("t2"))
        hib = {
            "t1": {"acc": True, "f1": True},
            "t2": {"acc": True, "f1": False},
        }
        _propagate_higher_is_better_([g], hib)
        assert hib["grp"]["acc"] is True
        assert hib["grp"]["f1"] is None

    def test_empty_groups_list(self):
        hib: dict = {"t": {"acc": True}}
        _propagate_higher_is_better_([], hib)
        # Nothing changes
        assert hib == {"t": {"acc": True}}


# ---------------------------------------------------------------------------
# TestToEvalResults
# ---------------------------------------------------------------------------


class TestToEvalResults:
    """Tests for _EvalAcc.to_eval_results()."""

    def _make_eval_acc(
        self, make_task, *, with_group: bool = False, has_aggregation: bool = True
    ):
        """Build a minimal _EvalAcc for testing to_eval_results()."""
        task = make_task(
            "t1",
            task_alias="Task One",
            num_fewshot=3,
            metadata={"version": 1},
            n_eval_docs=100,
        )
        acc_input = {
            "t1": make_result_acc(
                task,
                {("acc", "none"): [1.0, 0.0, 1.0]},
                agg={"acc": mean},
                hib={"acc": True},
            )
        }

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

        return _process_results(acc_input, groups=groups, bootstrap_iters=0)

    def test_output_has_required_keys(self, make_task):
        er = self._make_eval_acc(make_task)
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

    def test_results_contain_task_metrics(self, make_task):
        er = self._make_eval_acc(make_task)
        d = er._to_eval_results()
        assert "t1" in d["results"]
        assert "acc,none" in d["results"]["t1"]

    def test_n_samples_effective_from_sample_len(self, make_task):
        """Effective comes from sample_len (number of raw metric values)."""
        er = self._make_eval_acc(make_task)
        d = er._to_eval_results()
        assert d["n-samples"]["t1"]["original"] == 100
        # 3 raw metric values → sample_len == 3
        assert d["n-samples"]["t1"]["effective"] == 3

    def test_groups_key_present_when_group_has_aggregation(self, make_task):
        er = self._make_eval_acc(make_task, with_group=True, has_aggregation=True)
        d = er._to_eval_results()
        assert "groups" in d
        assert "grp" in d["groups"]

    def test_groups_key_absent_when_no_group_has_aggregation(self, make_task):
        er = self._make_eval_acc(make_task, with_group=True, has_aggregation=False)
        d = er._to_eval_results()
        assert "groups" not in d

    def test_groups_key_absent_when_no_groups(self, make_task):
        er = self._make_eval_acc(make_task, with_group=False)
        d = er._to_eval_results()
        assert "groups" not in d

    def test_samples_included_when_provided(self, make_task):
        er = self._make_eval_acc(make_task)
        d = er._to_eval_results(samples={"t1": [{"doc": 1}]})
        assert "samples" in d
        assert d["samples"]["t1"] == [{"doc": 1}]

    def test_samples_absent_when_not_provided(self, make_task):
        er = self._make_eval_acc(make_task)
        d = er._to_eval_results()
        assert "samples" not in d

    def test_higher_is_better_propagated_to_groups(self, make_task):
        er = self._make_eval_acc(make_task, with_group=True, has_aggregation=True)
        d = er._to_eval_results()
        assert "grp" in d["higher_is_better"]
        assert d["higher_is_better"]["grp"]["acc"] is True

    def test_configs_sorted(self, make_task):
        er = self._make_eval_acc(make_task)
        d = er._to_eval_results()
        assert list(d["configs"].keys()) == sorted(d["configs"].keys())

    def test_versions_sorted(self, make_task):
        er = self._make_eval_acc(make_task)
        d = er._to_eval_results()
        assert list(d["versions"].keys()) == sorted(d["versions"].keys())


# ---------------------------------------------------------------------------
# TestCollectResultsNSamples
# ---------------------------------------------------------------------------


class TestCollectResultsNSamples:
    """Tests for n_samples population via sample_len in collect_results()."""

    def test_n_samples_effective_equals_sample_len(self, make_task):
        t1 = make_task("t1", n_eval_docs=100)
        t2 = make_task("t2", n_eval_docs=200)
        accs = {
            "t1": make_result_acc(
                t1, {("acc", "none"): [1.0, 0.0, 1.0]}, agg={"acc": mean}
            ),
            "t2": make_result_acc(t2, {("acc", "none"): [0.0]}, agg={"acc": mean}),
        }
        result = _agg_and_collect(accs, bootstrap_iters=0)
        assert result.n_samples["t1"] == {"original": 100, "effective": 3}
        assert result.n_samples["t2"] == {"original": 200, "effective": 1}

    def test_n_samples_original_from_eval_docs(self, make_task):
        task = make_task("t1", n_eval_docs=42)
        accs = {
            "t1": make_result_acc(
                task, {("acc", "none"): [1.0, 0.5]}, agg={"acc": mean}
            )
        }
        result = _agg_and_collect(accs, bootstrap_iters=0)
        assert result.n_samples["t1"]["original"] == 42
        assert result.n_samples["t1"]["effective"] == 2


# ---------------------------------------------------------------------------
# TestScoredDoc
# ---------------------------------------------------------------------------


class TestScoredDoc:
    def test_frozen(self):
        sd = ScoredDoc(doc_id=0, reference="hello", scores={"acc": [1.0]})
        with pytest.raises(AttributeError):
            sd.doc_id = 1  # type: ignore[misc]

    def test_construction_single_repeat(self):
        sd = ScoredDoc(doc_id=0, reference="target", scores={"acc": [1.0]})
        assert sd.doc_id == 0
        assert sd.reference == "target"
        assert sd.scores == {"acc": [1.0]}

    def test_construction_multiple_repeats(self):
        sd = ScoredDoc(doc_id=5, reference="ref", scores={"acc": [1.0, 0.0, 1.0]})
        assert len(sd.scores["acc"]) == 3

    def test_construction_multiple_metrics(self):
        sd = ScoredDoc(
            doc_id=0,
            reference=[0, 1],
            scores={"acc": [0.5], "f1": [0.8]},
        )
        assert set(sd.scores.keys()) == {"acc", "f1"}


# ---------------------------------------------------------------------------
# TestMetricKey
# ---------------------------------------------------------------------------


class TestMetricKey:
    def test_str_basic(self):
        assert str(MetricKey("acc", "none")) == "acc,none"

    def test_str_stderr(self):
        assert str(MetricKey("acc", "none", is_stderr=True)) == "acc_stderr,none"

    def test_str_custom_scorer(self):
        assert str(MetricKey("f1", "strict")) == "f1,strict"

    def test_parse_basic(self):
        mk = MetricKey.parse("acc,none")
        assert mk is not None
        assert mk.metric == "acc"
        assert mk.scorer == "none"
        assert mk.is_stderr is False

    def test_parse_stderr(self):
        mk = MetricKey.parse("acc_stderr,none")
        assert mk is not None
        assert mk.metric == "acc"
        assert mk.scorer == "none"
        assert mk.is_stderr is True

    def test_parse_non_metric_key(self):
        assert MetricKey.parse("name") is None
        assert MetricKey.parse("alias") is None

    def test_parse_roundtrip(self):
        original = MetricKey("acc", "none")
        parsed = MetricKey.parse(str(original))
        assert parsed == original

    def test_parse_roundtrip_stderr(self):
        original = MetricKey("bleu", "exact", is_stderr=True)
        parsed = MetricKey.parse(str(original))
        assert parsed == original

    def test_parent_metric_plain(self):
        assert MetricKey("acc", "none").parent_metric is None

    def test_parent_metric_composite(self):
        mk = MetricKey("pass@1(exact_match)", "none")
        assert mk.parent_metric == "exact_match"

    def test_parent_metric_nested_parens(self):
        mk = MetricKey("sub(parent(inner))", "none")
        assert mk.parent_metric == "parent(inner)"

    def test_roundtrip_composite(self):
        original = MetricKey("pass@1(exact_match)", "none")
        parsed = MetricKey.parse(str(original))
        assert parsed == original


# ---------------------------------------------------------------------------
# TestScorerAggregationComposite
# ---------------------------------------------------------------------------


class TestScorerAggregationComposite:
    """Tests that Scorer.aggregate() and higher_is_better handle dict-returning reductions."""

    def _make_scorer_with_composite(self, parent_hib: bool = True):
        """Build a Scorer whose _reduced_docs contain composite keys from a dict reduction.

        Simulates what happens when a reduction like pass@k returns
        {"pass@1": 1.0, "pass@5": 0.5} and reduce() stores them as
        "pass@1(exact_match)" and "pass@5(exact_match)".
        """
        from lm_eval.api.metrics import Metric

        parent_metric = Metric(
            name="exact_match",
            fn=lambda *a, **kw: 0,
            aggregation=mean,
            higher_is_better=parent_hib,
        )
        scorer = Scorer(
            name="none",
            filter=_noop_filter(),
            metrics=[parent_metric],
        )
        scorer._reduced_docs = _reduced_docs_from_flat(
            {
                "pass@1(exact_match)": [1.0, 1.0, 0.0, 0.0],
                "pass@5(exact_match)": [0.5, 0.5, 0.5, 0.5],
            }
        )
        return scorer

    def test_aggregate_uses_parent_aggregation(self):
        scorer = self._make_scorer_with_composite()
        agg, sample_len = scorer.aggregate(scorer.reduced_docs, bootstrap_iters=0)
        assert agg["pass@1(exact_match),none"] == pytest.approx(0.5)
        assert agg["pass@5(exact_match),none"] == pytest.approx(0.5)
        assert sample_len == 4

    def test_aggregate_composite_stderr_present(self):
        scorer = self._make_scorer_with_composite()
        agg, _ = scorer.aggregate(scorer.reduced_docs, bootstrap_iters=100)
        assert "pass@1(exact_match)_stderr,none" in agg
        assert "pass@5(exact_match)_stderr,none" in agg

    def test_higher_is_better_includes_composite(self):
        scorer = self._make_scorer_with_composite(parent_hib=True)
        hib = scorer.higher_is_better
        assert hib["exact_match"] is True
        assert hib["pass@1(exact_match)"] is True
        assert hib["pass@5(exact_match)"] is True

    def test_higher_is_better_inherits_false(self):
        scorer = self._make_scorer_with_composite(parent_hib=False)
        hib = scorer.higher_is_better
        assert hib["exact_match"] is False
        assert hib["pass@1(exact_match)"] is False

    def test_higher_is_better_no_reduced_docs(self):
        """Without _reduced_docs, only base metrics appear."""
        from lm_eval.api.metrics import Metric

        parent_metric = Metric(
            name="exact_match",
            fn=lambda *a, **kw: 0,
            aggregation=mean,
            higher_is_better=True,
        )
        scorer = Scorer(name="none", filter=_noop_filter(), metrics=[parent_metric])
        hib = scorer.higher_is_better
        assert hib == {"exact_match": True}
