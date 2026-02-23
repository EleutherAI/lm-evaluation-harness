"""
Group model for organizing tasks into hierarchical collections.

A Group is a container for Tasks and/or sub-Groups that can compute
aggregated metrics across its members.

Example:
    >>> group = Group("mmlu")
    >>> group.add(mmlu_anatomy_task)
    >>> group.add(mmlu_biology_task)
    >>> all_tasks = group.get_all_tasks()  # [Task, Task]
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, cast

from typing_extensions import deprecated

from lm_eval.config.group import AggMetricConfig, GroupConfig


eval_logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from lm_eval.api.task import Task
    from lm_eval.result_schema import _TaskMetrics


@dataclass
class Group:
    """
    A Group is a container for Tasks and/or sub-Groups.

    Groups directly hold references to their children, making
    traversal and aggregation straightforward.

    Attributes:
        name: Unique identifier for this group (e.g., "mmlu", "mmlu::humanities")
        alias: Display name (defaults to name if not set)
        aggregate_metric_list: Optional list of metrics to aggregate across children
        metadata: Optional dict for user-defined metadata

    Example:
        >>> group = Group("mmlu")
        >>> group.add(anatomy_task)
        >>> group.add(biology_task)
        >>> group.get_all_tasks()  # [anatomy_task, biology_task]
    """

    name: str
    alias: str | None = None
    aggregate_metric_list: list[AggMetricConfig] | None = None
    metadata: dict[str, Any] | None = None
    _children: dict[str, Task | Group] = field(default_factory=dict, repr=False)
    _config: GroupConfig | None = field(default=None, repr=False)

    def add(self, item: Task | Group) -> None:
        """Add a task or subgroup to this group."""
        # Tasks have task_name, Groups have name
        key: str = cast(
            "str", item.task_name if hasattr(item, "task_name") else item.name
        )
        self._children[key] = item

    def pop(self, name: str) -> Group | Task | None:
        """Pop a child by name."""
        return self._children.pop(name, None)

    def get(self, name: str) -> Task | Group | None:
        """Get a child by name."""
        return self._children.get(name)

    def __contains__(self, name: str) -> bool:
        """Check if a child exists by name."""
        return name in self._children

    def __iter__(self):
        """Iterate over direct children (Task or Group objects)."""
        return iter(self._children.values())

    def __len__(self) -> int:
        """Number of direct children."""
        return len(self._children)

    # Query API

    def get_all_tasks(self, recursive: bool = True) -> list[Task]:
        """
        Get all leaf Task objects.

        Args:
            recursive: If True, include tasks from nested subgroups.
                       If False, only return direct Task children.

        Returns:
            List of Task objects (not Groups).
        """
        from lm_eval.api.task import Task

        tasks = []
        for item in self._children.values():
            if isinstance(item, Task):
                tasks.append(item)
            elif isinstance(item, Group) and recursive:
                tasks.extend(item.get_all_tasks(recursive=True))
        return tasks

    def get_all_groups(self, recursive: bool = True) -> list[Group]:
        """
        Get all subgroups.

        Args:
            recursive: If True, include nested subgroups.
                       If False, only return direct Group children.

        Returns:
            List of Group objects (not including self).
        """
        groups = []
        for item in self._children.values():
            if isinstance(item, Group):
                groups.append(item)
                if recursive:
                    groups.extend(item.get_all_groups(recursive=True))
        return groups

    @property
    def child_names(self) -> list[str]:
        """Names of direct children."""
        return list(self._children.keys())

    @property
    def version(self) -> str:
        """Version string from metadata, if available."""
        return "N/A" if not self.metadata else str(self.metadata.get("version", "N/A"))

    @property
    def has_aggregation(self) -> bool:
        """Whether this group defines aggregation metrics."""
        return (
            self.aggregate_metric_list is not None
            and len(self.aggregate_metric_list) > 0
        )

    def _discover_filters_for_metric(
        self, metric_name: str, task_metrics: dict[str, _TaskMetrics]
    ) -> list[str]:
        """
        Discover all filter names used with a specific metric in child tasks.

        Scans all leaf task metrics for keys matching "{metric},{filter}" pattern
        and returns unique filter names.

        Args:
            metric_name: Metric to search for (e.g., "acc", "acc_norm")
            task_metrics: Task metrics dict from EvalAcc.metrics

        Returns:
            Sorted list of unique filter names (e.g., ["custom", "none", "prefix"])
        """
        discovered_filters = set()
        leaf_tasks = [t.task_name for t in self.get_all_tasks()]

        for task_name in leaf_tasks:
            if task_name not in task_metrics:
                continue

            task_result = task_metrics[task_name]
            prefix = f"{metric_name},"

            for key in task_result.keys():
                # Look for "metric,filter" keys (exclude stderr keys)
                if key.startswith(prefix) and "_stderr" not in key:
                    filter_name = key[len(prefix) :]  # Extract filter part
                    discovered_filters.add(filter_name)

        return sorted(list(discovered_filters))  # Sort for deterministic ordering

    def aggregate(self, task_metrics: dict[str, _TaskMetrics]) -> _TaskMetrics:
        """
        Aggregate metrics for this group from its leaf task results.

        Args:
            task_metrics: {task_name: {metric_key: value, "sample_len": int, ...}}
                The full flat metrics dict (all tasks). This group only reads
                entries for its own leaf tasks (via ``get_all_tasks()``).

        Returns:
            Aggregated metrics dict for this group:
            {"alias": str, "acc,none": float, "acc_stderr,none": float, "sample_len": int, ...}
        """
        from lm_eval.api.metrics import aggregate_subtask_metrics, pooled_sample_stderr

        group_metrics: dict[str, Any] = {
            "alias": self.alias or self.name,
            "name": self.name,
        }

        if not self.aggregate_metric_list:
            return cast("_TaskMetrics", group_metrics)

        # Get leaf task names
        leaf_tasks = [t.task_name for t in self.get_all_tasks()]

        # group-level sample len. Not used for weighting, but useful metadata
        # Compute total sample_len once (across all leaf tasks), not per-filter
        group_metrics["sample_len"] = sum(
            task_metrics[name].get("sample_len", 0)
            for name in leaf_tasks
            if name in task_metrics
        )
        sample_count: dict[str, int] = {}

        for agg_config in self.aggregate_metric_list:
            # Determine filters: auto-discover if None, else use explicit list
            if agg_config.filter_list is None:
                filters_to_aggregate = self._discover_filters_for_metric(
                    agg_config.metric, task_metrics
                )
            else:
                filters_to_aggregate = agg_config.filter_list

            for filter_name in filters_to_aggregate:
                metric_key = f"{agg_config.metric},{filter_name}"
                stderr_key = f"{agg_config.metric}_stderr,{filter_name}"

                # Gather values from leaf tasks
                values: list[float] = []
                stderrs: list[float] = []
                sizes: list[int] = []
                tasks_with_metric: list[str] = []
                tasks_without_metric: list[str] = []

                for task_name in leaf_tasks:
                    if task_name not in task_metrics:
                        tasks_without_metric.append(task_name)
                        continue
                    task_result = task_metrics[task_name]
                    if metric_key in task_result:
                        values.append(task_result[metric_key])  # type:ignore[invalid-key]
                        sizes.append(task_result.get("sample_len", 0))
                        tasks_with_metric.append(task_name)
                        stderr_val = task_result.get(stderr_key)
                        if stderr_val is not None:
                            stderrs.append(stderr_val)
                    else:
                        tasks_without_metric.append(task_name)

                # Log warning if metric is missing in some tasks
                if values and tasks_without_metric:
                    eval_logger.warning(
                        f"Group '{self.name}': metric '{metric_key}' is missing in "
                        f"{len(tasks_without_metric)}/{len(leaf_tasks)} tasks. "
                        f"Missing in: {', '.join(tasks_without_metric[:5])}"
                        f"{f' and {len(tasks_without_metric) - 5} more' if len(tasks_without_metric) > 5 else ''}"
                    )

                if not values:
                    eval_logger.warning(
                        f"Group '{self.name}': no values found for metric '{metric_key}' across any tasks."
                    )

                if values:
                    group_metrics[metric_key] = aggregate_subtask_metrics(
                        values, sizes, agg_config.weight_by_size
                    )
                    sample_count[metric_key] = sum(sizes)

                    if len(stderrs) == len(values) and "N/A" not in stderrs:
                        group_metrics[stderr_key] = pooled_sample_stderr(stderrs, sizes)
                    else:
                        group_metrics[stderr_key] = "N/A"

        if sample_count:
            group_metrics["sample_count"] = sample_count

        return cast("_TaskMetrics", group_metrics)

    # I/O

    def to_dict(self) -> dict[str, Any] | None:
        """Convert to dictionary for serialization."""
        if self._config:
            return self._config.to_dict()
        result: dict[str, Any] = {"group": self.name}
        if self._children:
            result["task"] = list(self._children.keys())
        if self.alias:
            result["group_alias"] = self.alias
        if self.aggregate_metric_list:
            result["aggregate_metric_list"] = [
                asdict(agg) for agg in self.aggregate_metric_list
            ]
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_config(cls, config: GroupConfig | dict[str, Any]) -> Group:
        """
        Create a Group from a GroupConfig or raw dict (e.g., parsed from YAML).

        Note: This only creates the Group shell. Children must be added
        separately via group.add() after Tasks/subGroups are built.
        """
        if isinstance(config, dict):
            config = GroupConfig(**config)  # type:ignore[invalid-argument-type]

        return cls(
            name=config.group,
            alias=config.group_alias,
            aggregate_metric_list=cast(
                "list[AggMetricConfig]", config.aggregate_metric_list
            ),
            metadata=config.metadata,
            _config=config,
        )

    def __repr__(self):
        return f"Group(name={self.name!r}, len_children={len(self._children)}, children={self.child_names}, version={self.version})"


# =============================================================================
# Legacy compatibility - these will be deprecated
# =============================================================================


@deprecated("Use lm_eval.api.Group instead.")
class ConfigurableGroup(Group):
    """DEPRECATED: Use Group instead."""

    def __init__(self, config: dict | GroupConfig | None = None) -> None:
        self._config = (
            config if isinstance(config, GroupConfig) else GroupConfig(**(config or {}))
        )
        # Initialize Group dataclass fields so this acts as a proper wrapper
        self.name = self._config.group
        self.alias = self._config.group_alias
        self.aggregate_metric_list = cast(
            "list[AggMetricConfig]", self._config.aggregate_metric_list
        )
        self.metadata = self._config.metadata
        self._children = {}

    @property
    def group(self):
        return self.name

    @property
    def group_alias(self):
        return self.alias

    @property
    def version(self) -> str:
        if self._config and self._config.metadata:
            return str(self._config.metadata.get("version", "N/A"))
        return "N/A"

    @property
    def config(self):
        return self._config.to_dict() if self._config else None

    @property
    def group_name(self):
        return self._config.group if self._config else None

    @classmethod
    def from_group(cls, group: Group) -> ConfigurableGroup:
        """Wrap an existing Group as a ConfigurableGroup (backward compat)."""
        cg = object.__new__(cls)
        cg.__dict__.update(group.__dict__)
        if not cg._config:
            cg._config = GroupConfig(
                group=group.name,
                group_alias=group.alias,
                aggregate_metric_list=group.aggregate_metric_list,
                metadata=group.metadata,
            )
        return cg

    def __eq__(self, other):
        if isinstance(other, ConfigurableGroup):
            return self.group_name == other.group_name
        return NotImplemented

    def __hash__(self):
        return hash(self.group_name)

    def __repr__(self):
        return f"ConfigurableGroup(group={self.group},group_alias={self.group_alias})"
