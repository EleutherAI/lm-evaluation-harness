# type:ignore[invalid-assignment]
"""
Tests for Group class and filter auto-discovery functionality.
"""

from typing import TYPE_CHECKING

import pytest

from lm_eval.api.group import AggMetricConfig, Group
from lm_eval.api.task import Task


if TYPE_CHECKING:
    from lm_eval.result_schema import _TaskMetrics


class MockTask(Task):
    """Minimal mock task for testing."""

    VERSION = 0

    def __init__(self, task_name: str):
        self._task_name = task_name

    @property
    def task_name(self):
        return self._task_name

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return []

    def doc_to_text(self, doc):
        return ""

    def doc_to_target(self, doc):
        return ""

    def construct_requests(self, doc, ctx, **kwargs):
        return []

    def process_results(self, doc, results):
        return {}

    def aggregation(self):
        return {}

    def higher_is_better(self):
        return {}


class TestAggMetricConfig:
    """Tests for AggMetricConfig dataclass."""

    def test_default_filter_list_is_none(self):
        """Test that filter_list defaults to None for auto-discovery."""
        config = AggMetricConfig(metric="acc")
        assert config.filter_list is None

    def test_explicit_filter_list(self):
        """Test that explicit filter_list is preserved."""
        config = AggMetricConfig(metric="acc", filter_list=["none"])
        assert config.filter_list == ["none"]

    def test_string_filter_normalized_to_list(self):
        """Test that string filter_list is normalized to list."""
        config = AggMetricConfig(metric="acc", filter_list="none")  # type: ignore
        assert config.filter_list == ["none"]
        assert isinstance(config.filter_list, list)

    def test_empty_filter_list(self):
        """Test that empty filter_list is preserved."""
        config = AggMetricConfig(metric="acc", filter_list=[])
        assert config.filter_list == []

    def test_multiple_filters(self):
        """Test multiple filters in list."""
        config = AggMetricConfig(metric="acc", filter_list=["none", "prefix", "custom"])
        assert config.filter_list == ["none", "prefix", "custom"]

    def test_default_aggregation_is_mean(self):
        """Test that aggregation defaults to 'mean'."""
        config = AggMetricConfig(metric="acc")
        assert config.aggregation == "mean"

    def test_default_weight_by_size_is_true(self):
        """Test that weight_by_size defaults to True."""
        config = AggMetricConfig(metric="acc")
        assert config.weight_by_size is True


class TestGroupFilterDiscovery:
    """Tests for filter auto-discovery in Group.aggregate()."""

    def setup_method(self):
        """Set up common test fixtures."""
        # Create mock tasks
        self.task_a = MockTask("task_a")
        self.task_b = MockTask("task_b")

        # Create task metrics with multiple filters
        self.task_metrics: dict[str, _TaskMetrics] = {
            "task_a": {
                "name": "Task A",
                "alias": "Task A",
                "sample_len": 100,
                "acc_norm,none": 0.85,
                "acc_norm_stderr,none": 0.02,
                "acc_norm,prefix": 0.88,
                "acc_norm_stderr,prefix": 0.015,
            },  # ty:ignore[invalid-key]
            "task_b": {
                "name": "Task B",
                "alias": "Task B",
                "sample_len": 150,
                "acc_norm,none": 0.90,
                "acc_norm_stderr,none": 0.018,
                "acc_norm,custom": 0.92,
                "acc_norm_stderr,custom": 0.012,
            },  # ty:ignore[invalid-key]
        }

    def test_discover_filters_single_filter(self):
        """Test discovering filters when all tasks have the same filter."""
        group = Group(name="test_group")
        group.add(self.task_a)

        metrics = {
            "task_a": {
                "acc,none": 0.85,
                "acc_stderr,none": 0.02,
            }
        }

        discovered = group._discover_filters_for_metric("acc", metrics)
        assert discovered == ["none"]

    def test_discover_filters_multiple_filters(self):
        """Test discovering multiple filters across tasks."""
        group = Group(name="test_group")
        group.add(self.task_a)
        group.add(self.task_b)

        discovered = group._discover_filters_for_metric("acc_norm", self.task_metrics)
        assert discovered == ["custom", "none", "prefix"]  # Sorted alphabetically

    def test_discover_filters_no_matches(self):
        """Test discovering filters when metric doesn't exist."""
        group = Group(name="test_group")
        group.add(self.task_a)

        discovered = group._discover_filters_for_metric(
            "nonexistent_metric", self.task_metrics
        )
        assert discovered == []

    def test_discover_filters_excludes_stderr(self):
        """Test that stderr keys are excluded from discovery."""
        group = Group(name="test_group")
        group.add(self.task_a)

        metrics: dict[str, _TaskMetrics] = {
            "task_a": {
                "name": "Task A",
                "alias": "Task A",
                "sample_len": 100,
                "acc,none": 0.85,
                "acc_stderr,none": 0.02,  # Should be excluded
            }
        }  # ty:ignore[invalid-assignment]

        discovered = group._discover_filters_for_metric("acc", metrics)
        # Only "none" from "acc,none", not from "acc_stderr,none"
        assert discovered == ["none"]

    def test_discover_filters_partial_availability(self):
        """Test discovery when filters are available in some but not all tasks."""
        group = Group(name="test_group")
        group.add(self.task_a)
        group.add(self.task_b)

        # task_a has "none" and "prefix", task_b has "none" and "custom"
        discovered = group._discover_filters_for_metric("acc_norm", self.task_metrics)
        assert set(discovered) == {"none", "prefix", "custom"}


class TestGroupAggregation:
    """Tests for Group.aggregate() with filter auto-discovery."""

    def setup_method(self):
        """Set up common test fixtures."""
        self.task_a = MockTask("task_a")
        self.task_b = MockTask("task_b")

        self.task_metrics: dict[str, _TaskMetrics] = {
            "task_a": {
                "name": "Task A",
                "alias": "Task A",
                "sample_len": 100,
                "acc_norm,none": 0.80,
                "acc_norm_stderr,none": 0.02,
                "acc_norm,prefix": 0.85,
                "acc_norm_stderr,prefix": 0.015,
            },
            "task_b": {
                "name": "Task B",
                "alias": "Task B",
                "sample_len": 100,
                "acc_norm,none": 0.90,
                "acc_norm_stderr,none": 0.018,
                "acc_norm,custom": 0.92,
                "acc_norm_stderr,custom": 0.012,
            },
        }

    def test_auto_discovery_aggregates_all_filters(self):
        """Test that auto-discovery (filter_list=None) aggregates all filters."""
        group = Group(
            name="test_group",
            aggregate_metric_list=[
                AggMetricConfig(metric="acc_norm")
            ],  # filter_list=None by default
        )
        group.add(self.task_a)
        group.add(self.task_b)

        result = group.aggregate(self.task_metrics)

        # Should have all three filters aggregated
        assert "acc_norm,none" in result
        assert "acc_norm,prefix" in result
        assert "acc_norm,custom" in result

        # Verify values are computed correctly
        # "none" appears in both tasks: (0.80 + 0.90) / 2 = 0.85
        assert result["acc_norm,none"] == pytest.approx(0.85)

        # "prefix" only in task_a
        assert result["acc_norm,prefix"] == 0.85

        # "custom" only in task_b
        assert result["acc_norm,custom"] == 0.92

    def test_explicit_filter_list_backward_compatibility(self):
        """Test that explicit filter_list only aggregates specified filters."""
        group = Group(
            name="test_group",
            aggregate_metric_list=[
                AggMetricConfig(metric="acc_norm", filter_list=["none"])
            ],
        )
        group.add(self.task_a)
        group.add(self.task_b)

        result = group.aggregate(self.task_metrics)

        # Should only have "none" filter
        assert "acc_norm,none" in result
        assert "acc_norm,prefix" not in result
        assert "acc_norm,custom" not in result

    def test_multiple_explicit_filters(self):
        """Test aggregation with multiple explicit filters."""
        group = Group(
            name="test_group",
            aggregate_metric_list=[
                AggMetricConfig(metric="acc_norm", filter_list=["none", "prefix"])
            ],
        )
        group.add(self.task_a)
        group.add(self.task_b)

        result = group.aggregate(self.task_metrics)

        # Should have "none" and "prefix", but not "custom"
        assert "acc_norm,none" in result
        assert "acc_norm,prefix" in result
        assert "acc_norm,custom" not in result

    def test_empty_filter_list_no_aggregation(self):
        """Test that empty filter_list results in no metric aggregation."""
        group = Group(
            name="test_group",
            aggregate_metric_list=[AggMetricConfig(metric="acc_norm", filter_list=[])],
        )
        group.add(self.task_a)

        result = group.aggregate(self.task_metrics)

        # Should only have alias, name, and sample_len — no metric keys
        assert "alias" in result
        assert "name" in result
        assert "acc_norm,none" not in result
        assert (
            len([k for k in result.keys() if k not in ("alias", "name", "sample_len")])
            == 0
        )

    def test_multiple_metrics_auto_discovery(self):
        """Test auto-discovery with multiple metrics."""
        # Add another metric to test data
        metrics_extended = {
            "task_a": {
                **self.task_metrics["task_a"],
                "acc,none": 0.75,
                "acc_stderr,none": 0.03,
            },
            "task_b": {
                **self.task_metrics["task_b"],
                "acc,none": 0.85,
                "acc_stderr,none": 0.025,
            },
        }

        group = Group(
            name="test_group",
            aggregate_metric_list=[
                AggMetricConfig(metric="acc_norm"),  # Auto-discover
                AggMetricConfig(metric="acc"),  # Auto-discover
            ],
        )
        group.add(self.task_a)
        group.add(self.task_b)

        result = group.aggregate(metrics_extended)

        # Should have acc_norm with all its filters
        assert "acc_norm,none" in result
        assert "acc_norm,prefix" in result
        assert "acc_norm,custom" in result

        # Should have acc with its filter
        assert "acc,none" in result

    def test_mixed_auto_and_explicit_filters(self):
        """Test using auto-discovery for one metric and explicit for another."""
        metrics_extended = {
            "task_a": {
                **self.task_metrics["task_a"],
                "acc,none": 0.75,
                "acc,prefix": 0.78,
            },
            "task_b": {
                **self.task_metrics["task_b"],
                "acc,none": 0.85,
            },
        }

        group = Group(
            name="test_group",
            aggregate_metric_list=[
                AggMetricConfig(metric="acc_norm"),  # Auto-discover
                AggMetricConfig(metric="acc", filter_list=["none"]),  # Explicit
            ],
        )
        group.add(self.task_a)
        group.add(self.task_b)

        result = group.aggregate(metrics_extended)

        # acc_norm should have all filters (auto-discovered)
        assert "acc_norm,none" in result
        assert "acc_norm,prefix" in result
        assert "acc_norm,custom" in result

        # acc should only have "none" (explicit)
        assert "acc,none" in result
        assert "acc,prefix" not in result

    def test_stderr_aggregation_with_auto_discovery(self):
        """Test that stderr values are properly aggregated with auto-discovery."""
        group = Group(
            name="test_group",
            aggregate_metric_list=[AggMetricConfig(metric="acc_norm")],
        )
        group.add(self.task_a)
        group.add(self.task_b)

        result = group.aggregate(self.task_metrics)

        # Should have stderr for all discovered filters
        assert "acc_norm_stderr,none" in result
        assert "acc_norm_stderr,prefix" in result
        assert "acc_norm_stderr,custom" in result

        # Stderr for "none" should be computed (both tasks have it)
        assert result["acc_norm_stderr,none"] != "N/A"

        # Stderr for "prefix" should be from task_a only
        assert result["acc_norm_stderr,prefix"] == 0.015

        # Stderr for "custom" should be from task_b only
        assert result["acc_norm_stderr,custom"] == 0.012

    def test_sample_len_count_with_auto_discovery(self):
        """
        Test that sample_len is the total across all leaf tasks, and
        per-metric sample_count reflects contributing tasks.
        """
        group = Group(
            name="test_group",
            aggregate_metric_list=[AggMetricConfig(metric="acc_norm")],
        )
        group.add(self.task_a)
        group.add(self.task_b)

        result = group.aggregate(self.task_metrics)

        # sample_len is total across ALL leaf tasks (100 + 100)
        assert result["sample_len"] == 200
        # per-metric sample_count: "none" appears in both tasks
        assert result["sample_count"]["acc_norm,none"] == 200
        # "prefix" only in task_a (100), "custom" only in task_b (100)
        assert result["sample_count"]["acc_norm,prefix"] == 100
        assert result["sample_count"]["acc_norm,custom"] == 100

    def test_sample_count_per_metric_with_asymmetric_filters(self):
        """Per-metric sample_count reflects contributing tasks; sample_len is the total."""
        metrics = {
            "task_a": {"sample_len": 100, "acc,none": 0.8, "acc,prefix": 0.85},
            "task_b": {"sample_len": 200, "acc,none": 0.9},  # no prefix
        }
        group = Group(name="g", aggregate_metric_list=[AggMetricConfig(metric="acc")])
        group.add(MockTask("task_a"))
        group.add(MockTask("task_b"))
        result = group.aggregate(metrics)
        assert result["sample_len"] == 300  # total across all leaf tasks
        assert result["sample_count"]["acc,none"] == 300  # both tasks contribute
        assert result["sample_count"]["acc,prefix"] == 100  # only task_a contributes


class TestGroupWeightedAggregation:
    """Test weighted aggregation with filter auto-discovery."""

    def test_weighted_aggregation_auto_discovery(self):
        """Test that weight_by_size works correctly with auto-discovery."""
        task_a = MockTask("task_a")
        task_b = MockTask("task_b")

        # task_a has 100 sample_len, task_b has 200 sample_len
        metrics = {
            "task_a": {
                "sample_len": 100,
                "acc,none": 0.60,
                "acc_stderr,none": 0.02,
            },
            "task_b": {
                "sample_len": 200,
                "acc,none": 0.90,
                "acc_stderr,none": 0.01,
            },
        }

        # Test with weight_by_size=True (default)
        group_weighted = Group(
            name="test_weighted",
            aggregate_metric_list=[AggMetricConfig(metric="acc", weight_by_size=True)],
        )
        group_weighted.add(task_a)
        group_weighted.add(task_b)

        result_weighted = group_weighted.aggregate(metrics)

        # Weighted average: (0.60 * 100 + 0.90 * 200) / (100 + 200) = (60 + 180) / 300 = 0.80
        assert result_weighted["acc,none"] == pytest.approx(0.80)

        # Test with weight_by_size=False
        group_unweighted = Group(
            name="test_unweighted",
            aggregate_metric_list=[AggMetricConfig(metric="acc", weight_by_size=False)],
        )
        group_unweighted.add(task_a)
        group_unweighted.add(task_b)

        result_unweighted = group_unweighted.aggregate(metrics)

        # Unweighted average: (0.60 + 0.90) / 2 = 0.75
        assert result_unweighted["acc,none"] == pytest.approx(0.75)


class TestGroupEdgeCases:
    """Test edge cases for Group aggregation."""

    def test_no_aggregation_config(self):
        """Test group with no aggregation config."""
        group = Group(name="test_group")
        task = MockTask("task_a")
        group.add(task)

        result = group.aggregate({"task_a": {"acc,none": 0.85}})

        # Should only return alias and name
        assert result == {"alias": "test_group", "name": "test_group"}

    def test_task_not_in_metrics(self):
        """Test when a task is in the group but not in metrics dict."""
        group = Group(
            name="test_group", aggregate_metric_list=[AggMetricConfig(metric="acc")]
        )
        task_a = MockTask("task_a")
        task_b = MockTask("task_b")
        group.add(task_a)
        group.add(task_b)

        # Only task_a in metrics
        metrics = {
            "task_a": {
                "sample_len": 100,
                "acc,none": 0.85,
            }
        }

        result = group.aggregate(metrics)

        # Should still work, using only task_a
        assert "acc,none" in result
        assert result["acc,none"] == 0.85

    def test_metric_missing_in_some_tasks(self, caplog):
        """Test when a metric is missing in some tasks."""
        import logging

        group = Group(
            name="test_group", aggregate_metric_list=[AggMetricConfig(metric="acc")]
        )
        task_a = MockTask("task_a")
        task_b = MockTask("task_b")
        group.add(task_a)
        group.add(task_b)

        metrics = {
            "task_a": {
                "sample_len": 100,
                "acc,none": 0.85,
                "acc,prefix": 0.90,
            },
            "task_b": {
                "sample_len": 150,
                # Missing "acc" metrics entirely
            },
        }

        with caplog.at_level(logging.WARNING):
            result = group.aggregate(metrics)

        # Should aggregate only from task_a
        assert "acc,none" in result
        assert "acc,prefix" in result
        assert result["acc,none"] == 0.85
        assert result["acc,prefix"] == 0.90

        # Should log warnings for missing metrics
        assert "acc,none" in caplog.text
        assert "missing" in caplog.text.lower()
        assert "task_b" in caplog.text

        # Verify we got warnings for both filter variants
        warning_messages = [
            rec.message for rec in caplog.records if rec.levelname == "WARNING"
        ]
        assert len(warning_messages) == 2  # One for "none", one for "prefix"


class TestGroup:
    """Tests for Group core container API."""

    def setup_method(self):
        self.task_a = MockTask("task_a")
        self.task_b = MockTask("task_b")

    # --- add / remove / get ---

    def test_add_task_uses_task_name(self):
        group = Group(name="g")
        group.add(self.task_a)
        assert "task_a" in group
        assert group.get("task_a") is self.task_a

    def test_add_group_uses_name(self):
        parent = Group(name="parent")
        child = Group(name="child")
        parent.add(child)
        assert "child" in parent
        assert parent.get("child") is child

    def test_pop_existing_child(self):
        group = Group(name="g")
        group.add(self.task_a)
        child = group.pop("task_a")
        assert child is self.task_a
        assert "task_a" not in group

    def test_pop_nonexistent_child_no_error(self):
        group = Group(name="g")
        child = group.pop("nonexistent")  # should not raise
        assert child is None

    def test_get_existing(self):
        group = Group(name="g")
        group.add(self.task_a)
        assert group.get("task_a") is self.task_a

    def test_get_missing_returns_none(self):
        group = Group(name="g")
        assert group.get("missing") is None

    # --- __contains__ ---

    def test_contains_present(self):
        group = Group(name="g")
        group.add(self.task_a)
        assert "task_a" in group

    def test_contains_absent(self):
        group = Group(name="g")
        assert "task_a" not in group

    # --- __iter__ ---

    def test_iter_yields_child_values(self):
        group = Group(name="g")
        group.add(self.task_a)
        group.add(self.task_b)
        children = list(group)
        assert len(children) == 2
        assert self.task_a in children
        assert self.task_b in children

    # --- __len__ ---

    def test_len(self):
        group = Group(name="g")
        assert len(group) == 0
        group.add(self.task_a)
        assert len(group) == 1
        group.add(self.task_b)
        assert len(group) == 2

    # --- get_all_tasks ---

    def test_get_all_tasks_recursive(self):
        parent = Group(name="parent")
        child_group = Group(name="child")
        child_group.add(self.task_a)
        parent.add(child_group)
        parent.add(self.task_b)

        tasks = parent.get_all_tasks(recursive=True)
        assert len(tasks) == 2
        assert self.task_a in tasks
        assert self.task_b in tasks

    def test_get_all_tasks_non_recursive(self):
        parent = Group(name="parent")
        child_group = Group(name="child")
        child_group.add(self.task_a)
        parent.add(child_group)
        parent.add(self.task_b)

        tasks = parent.get_all_tasks(recursive=False)
        assert tasks == [self.task_b]

    # --- get_all_groups ---

    def test_get_all_groups_recursive(self):
        grandchild = Group(name="grandchild")
        child = Group(name="child")
        child.add(grandchild)
        parent = Group(name="parent")
        parent.add(child)

        groups = parent.get_all_groups(recursive=True)
        assert len(groups) == 2
        assert child in groups
        assert grandchild in groups

    def test_get_all_groups_non_recursive(self):
        grandchild = Group(name="grandchild")
        child = Group(name="child")
        child.add(grandchild)
        parent = Group(name="parent")
        parent.add(child)

        groups = parent.get_all_groups(recursive=False)
        assert groups == [child]

    # --- children property ---

    def test_child_names_returns_keys(self):
        group = Group(name="g")
        group.add(self.task_a)
        group.add(self.task_b)
        assert group.child_names == ["task_a", "task_b"]

    # --- has_aggregation ---

    def test_has_aggregation_true(self):
        group = Group(
            name="g",
            aggregate_metric_list=[AggMetricConfig(metric="acc")],
        )
        assert group.has_aggregation is True

    def test_has_aggregation_false_none(self):
        group = Group(name="g")
        assert group.has_aggregation is False

    def test_has_aggregation_false_empty(self):
        group = Group(name="g", aggregate_metric_list=[])
        assert group.has_aggregation is False

    # --- __repr__ ---

    def test_repr(self):
        group = Group(name="test_group")
        group.add(self.task_a)
        group.add(self.task_b)
        r = repr(group)
        assert "test_group" in r
        assert "2" in r


class TestGroupSerialization:
    """Tests for Group serialization and deserialization."""

    def test_to_dict_round_trip(self):
        group = Group(
            name="mmlu",
            alias="MMLU",
            aggregate_metric_list=[AggMetricConfig(metric="acc", filter_list=["none"])],
            metadata={"version": 1},
        )
        task = MockTask("task_a")
        group.add(task)

        d = group.to_dict()
        assert d["group"] == "mmlu"
        assert d["task"] == ["task_a"]
        assert d["group_alias"] == "MMLU"
        assert len(d["aggregate_metric_list"]) == 1
        assert d["aggregate_metric_list"][0]["metric"] == "acc"
        assert d["metadata"] == {"version": 1}

    def test_from_config_basic(self):
        config = {
            "group": "my_group",
            "group_alias": "My Group",
            "aggregate_metric_list": [
                {
                    "metric": "acc",
                    "filter_list": ["none"],
                    "aggregation": "mean",
                    "weight_by_size": True,
                }
            ],
            "metadata": {"version": 2},
        }
        group = Group.from_config(config)
        assert group.name == "my_group"
        assert group.alias == "My Group"
        assert group.aggregate_metric_list
        assert len(group.aggregate_metric_list) == 1
        assert group.aggregate_metric_list[0].metric == "acc"
        assert group.metadata == {"version": 2}

    def test_from_config_single_dict_agg_metric(self):
        """aggregate_metric_list given as a single dict (not wrapped in a list)."""
        config = {
            "group": "g",
            "aggregate_metric_list": {"metric": "acc"},
        }
        group = Group.from_config(config)
        assert group.aggregate_metric_list
        assert len(group.aggregate_metric_list) == 1
        assert group.aggregate_metric_list[0].metric == "acc"

    def test_from_config_missing_group_key(self):
        """Missing 'group' key raise error."""
        config = {}
        with pytest.raises(TypeError):
            Group.from_config(config)

    def test_to_dict_no_optional_fields(self):
        """to_dict omits optional keys when not set."""
        group = Group(name="bare")
        d = group.to_dict()
        assert d == {"group": "bare"}
        assert "task" not in d
        assert "group_alias" not in d
        assert "aggregate_metric_list" not in d
        assert "metadata" not in d


class TestAggMetricConfigValidation:
    """Tests for AggMetricConfig validation."""

    def test_invalid_aggregation_raises(self):
        with pytest.raises(ValueError, match="only pre-defined aggregation"):
            AggMetricConfig(metric="acc", aggregation="median")

    def test_callable_aggregation_allowed(self):
        config = AggMetricConfig(metric="acc", aggregation=max)
        assert config.aggregation is max
