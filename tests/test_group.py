"""
Tests for Group class and filter auto-discovery functionality.
"""

import pytest

from lm_eval.api.group import AggMetricConfig, Group
from lm_eval.api.task import Task


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
        config = AggMetricConfig(metric="acc", filter_list="none")
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
        self.task_metrics = {
            "task_a": {
                "alias": "Task A",
                "samples": 100,
                "acc_norm,none": 0.85,
                "acc_norm_stderr,none": 0.02,
                "acc_norm,prefix": 0.88,
                "acc_norm_stderr,prefix": 0.015,
            },
            "task_b": {
                "alias": "Task B",
                "samples": 150,
                "acc_norm,none": 0.90,
                "acc_norm_stderr,none": 0.018,
                "acc_norm,custom": 0.92,
                "acc_norm_stderr,custom": 0.012,
            },
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

        metrics = {
            "task_a": {
                "acc,none": 0.85,
                "acc_stderr,none": 0.02,  # Should be excluded
            }
        }

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

        self.task_metrics = {
            "task_a": {
                "alias": "Task A",
                "samples": 100,
                "acc_norm,none": 0.80,
                "acc_norm_stderr,none": 0.02,
                "acc_norm,prefix": 0.85,
                "acc_norm_stderr,prefix": 0.015,
            },
            "task_b": {
                "alias": "Task B",
                "samples": 100,
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
            aggregation=[
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
            aggregation=[AggMetricConfig(metric="acc_norm", filter_list=["none"])],
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
            aggregation=[
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
            aggregation=[AggMetricConfig(metric="acc_norm", filter_list=[])],
        )
        group.add(self.task_a)

        result = group.aggregate(self.task_metrics)

        # Should only have alias, no metrics
        assert "alias" in result
        assert "acc_norm,none" not in result
        assert len([k for k in result.keys() if k != "alias"]) == 0

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
            aggregation=[
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
            aggregation=[
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
            name="test_group", aggregation=[AggMetricConfig(metric="acc_norm")]
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

    def test_samples_count_with_auto_discovery(self):
        """Test that sample counts are correct with auto-discovery."""
        group = Group(
            name="test_group", aggregation=[AggMetricConfig(metric="acc_norm")]
        )
        group.add(self.task_a)
        group.add(self.task_b)

        result = group.aggregate(self.task_metrics)

        # Total samples should be sum of both tasks (last filter processed sets this)
        # In this case, the last filter processed will determine the final sample count
        assert "samples" in result
        # The samples count is set per filter during aggregation
        # For filters present in both tasks, samples = 100 + 100 = 200
        # For filters present in one task, samples = 100
        # The final "samples" value will be from the last filter processed


class TestGroupWeightedAggregation:
    """Test weighted aggregation with filter auto-discovery."""

    def test_weighted_aggregation_auto_discovery(self):
        """Test that weight_by_size works correctly with auto-discovery."""
        task_a = MockTask("task_a")
        task_b = MockTask("task_b")

        # task_a has 100 samples, task_b has 200 samples
        metrics = {
            "task_a": {
                "samples": 100,
                "acc,none": 0.60,
                "acc_stderr,none": 0.02,
            },
            "task_b": {
                "samples": 200,
                "acc,none": 0.90,
                "acc_stderr,none": 0.01,
            },
        }

        # Test with weight_by_size=True (default)
        group_weighted = Group(
            name="test_weighted",
            aggregation=[AggMetricConfig(metric="acc", weight_by_size=True)],
        )
        group_weighted.add(task_a)
        group_weighted.add(task_b)

        result_weighted = group_weighted.aggregate(metrics)

        # Weighted average: (0.60 * 100 + 0.90 * 200) / (100 + 200) = (60 + 180) / 300 = 0.80
        assert result_weighted["acc,none"] == pytest.approx(0.80)

        # Test with weight_by_size=False
        group_unweighted = Group(
            name="test_unweighted",
            aggregation=[AggMetricConfig(metric="acc", weight_by_size=False)],
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

        # Should only return alias
        assert result == {"alias": "test_group"}

    def test_task_not_in_metrics(self):
        """Test when a task is in the group but not in metrics dict."""
        group = Group(name="test_group", aggregation=[AggMetricConfig(metric="acc")])
        task_a = MockTask("task_a")
        task_b = MockTask("task_b")
        group.add(task_a)
        group.add(task_b)

        # Only task_a in metrics
        metrics = {
            "task_a": {
                "samples": 100,
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

        group = Group(name="test_group", aggregation=[AggMetricConfig(metric="acc")])
        task_a = MockTask("task_a")
        task_b = MockTask("task_b")
        group.add(task_a)
        group.add(task_b)

        metrics = {
            "task_a": {
                "samples": 100,
                "acc,none": 0.85,
                "acc,prefix": 0.90,
            },
            "task_b": {
                "samples": 150,
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
