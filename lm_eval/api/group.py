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

import abc
from dataclasses import asdict, dataclass, field
from inspect import getsource
from typing import TYPE_CHECKING, Any, cast

from typing_extensions import deprecated


if TYPE_CHECKING:
    from collections.abc import Callable

    from lm_eval.api.task import Task


@dataclass
class AggMetricConfig:
    """Configuration for how to aggregate a metric across a group's children."""

    metric: str
    filter_list: list[str] = field(default_factory=lambda: ["none"])
    aggregation: str | Callable = "mean"
    weight_by_size: bool = True

    def __post_init__(self):
        if self.aggregation != "mean" and not callable(self.aggregation):
            raise ValueError(
                f"Currently, 'mean' is the only pre-defined aggregation. Got '{self.aggregation}'."
            )
        # Normalize filter_list to always be a list
        if isinstance(self.filter_list, str):
            self.filter_list = [self.filter_list]


@dataclass
class Group:
    """
    A Group is a container for Tasks and/or sub-Groups.

    Groups directly hold their children (not just string references), making
    traversal and aggregation straightforward.

    Attributes:
        name: Unique identifier for this group (e.g., "mmlu", "mmlu::humanities")
        alias: Display name (defaults to name if not set)
        aggregation: Optional list of metrics to aggregate across children
        metadata: Optional dict for user-defined metadata

    Example:
        >>> group = Group("mmlu")
        >>> group.add(anatomy_task)
        >>> group.add(biology_task)
        >>> group.get_all_tasks()  # [anatomy_task, biology_task]
    """

    name: str
    alias: str | None = None
    aggregation: list[AggMetricConfig] | None = None
    metadata: dict[str, Any] | None = None
    _children: dict[str, Task | Group] = field(default_factory=dict, repr=False)

    def add(self, item: Task | Group) -> None:
        """Add a task or subgroup to this group."""
        # Tasks have task_name, Groups have name
        key: str = cast(
            "str", item.task_name if hasattr(item, "task_name") else item.name
        )
        self._children[key] = item

    def remove(self, name: str) -> None:
        """Remove a child by name."""
        self._children.pop(name, None)

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
    def children(self) -> list[str]:
        """Names of direct children (for backward compatibility)."""
        return list(self._children.keys())

    @property
    def display_name(self) -> str:
        """Name to show in output (alias if set, otherwise name)."""
        return self.alias or self.name

    @property
    def has_aggregation(self) -> bool:
        """Whether this group defines aggregation metrics."""
        return self.aggregation is not None and len(self.aggregation) > 0

    def aggregate(self, task_metrics: dict[str, dict]) -> dict[str, Any]:
        """
        Aggregate metrics for this group from its leaf task results.

        Args:
            task_metrics: {task_name: {metric_key: value, "samples": int, ...}}
                Results from leaf tasks (EvalResults.metrics)

        Returns:
            Aggregated metrics dict for this group:
            {"alias": str, "acc,none": float, "acc_stderr,none": float, "samples": int, ...}
        """
        from lm_eval.api.metrics import aggregate_subtask_metrics, pooled_sample_stderr

        group_metrics: dict[str, Any] = {"alias": self.display_name}

        if not self.aggregation:
            return group_metrics

        # Get leaf task names
        leaf_tasks = [t.task_name for t in self.get_all_tasks()]

        for agg_config in self.aggregation:
            for filter_name in agg_config.filter_list:
                metric_key = f"{agg_config.metric},{filter_name}"
                stderr_key = f"{agg_config.metric}_stderr,{filter_name}"

                # Gather values from leaf tasks
                values: list[float] = []
                stderrs: list[float] = []
                sizes: list[int] = []

                for task_name in leaf_tasks:
                    if task_name not in task_metrics:
                        continue
                    task_result = task_metrics[task_name]
                    if metric_key in task_result:
                        values.append(task_result[metric_key])
                        sizes.append(task_result.get("samples", 0))
                        stderr_val = task_result.get(stderr_key)
                        if stderr_val is not None:
                            stderrs.append(stderr_val)

                if values:
                    group_metrics[metric_key] = aggregate_subtask_metrics(
                        values, sizes, agg_config.weight_by_size
                    )
                    group_metrics["samples"] = sum(sizes)

                    if len(stderrs) == len(values) and "N/A" not in stderrs:
                        group_metrics[stderr_key] = pooled_sample_stderr(stderrs, sizes)
                    else:
                        group_metrics[stderr_key] = "N/A"

        return group_metrics

    # I/O

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {"group": self.name}
        if self._children:
            result["children"] = list(self._children.keys())
        if self.alias:
            result["group_alias"] = self.alias
        if self.aggregation:
            result["aggregate_metric_list"] = [asdict(agg) for agg in self.aggregation]
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Group:
        """
        Create a Group from a config dict (e.g., parsed from YAML).

        Note: This only creates the Group shell. Children must be added
        separately via group.add() after Tasks/subGroups are built.
        """
        name: str = config.get("group", "")

        # Parse aggregation config
        aggregation = None
        if agg_list := config.get("aggregate_metric_list"):
            if isinstance(agg_list, dict):
                agg_list = [agg_list]
            aggregation = [
                AggMetricConfig(**item) if isinstance(item, dict) else item
                for item in agg_list
            ]

        return cls(
            name=name,
            alias=config.get("group_alias"),
            aggregation=aggregation,
            metadata=config.get("metadata"),
        )

    def __repr__(self):
        return f"Group(name={self.name!r}, children={len(self._children)})"


# =============================================================================
# Legacy compatibility - these will be deprecated
# =============================================================================


@deprecated("Use `Group` instead.")
@dataclass
class GroupConfig(dict):
    """DEPRECATED: Use Group instead."""

    group: str | None = None
    group_alias: str | None = None
    task: str | list | None = None
    aggregate_metric_list: (
        list[AggMetricConfig] | AggMetricConfig | dict[str, str] | None
    ) = None
    metadata: dict | None = None

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, item, value):
        return setattr(self, item, value)

    def __post_init__(self):
        if self.aggregate_metric_list is not None:
            if isinstance(self.aggregate_metric_list, dict):
                self.aggregate_metric_list = [self.aggregate_metric_list]
            self.aggregate_metric_list = [
                AggMetricConfig(**item) if isinstance(item, dict) else item
                for item in self.aggregate_metric_list
            ]

    def to_dict(self, keep_callable: bool = False) -> dict:
        cfg_dict = asdict(self)
        for k, v in list(cfg_dict.items()):
            if callable(v):
                cfg_dict[k] = self.serialize_function(v, keep_callable=keep_callable)
        return cfg_dict

    def serialize_function(
        self, value: Callable | str, keep_callable=False
    ) -> Callable | str:
        if keep_callable:
            return value
        try:
            return getsource(value)
        except (TypeError, OSError):
            return str(value)


@deprecated("Use Group instead.")
class ConfigurableGroup(abc.ABC):  # noqa: B024
    """DEPRECATED: Use Group instead."""

    def __init__(self, config: dict | None = None) -> None:
        self._config = GroupConfig(**config)
        # Also create a new-style Group for forward compatibility
        self._group = Group.from_config(config or {})

    @property
    def group(self):
        return self._config.group

    @property
    def group_alias(self):
        return self._config.group_alias

    @property
    def version(self):
        return self._config.metadata.get("version") if self._config.metadata else None

    @property
    def config(self):
        return self._config.to_dict()

    @property
    def group_name(self) -> Any:
        return self._config.group

    def __repr__(self):
        return f"ConfigurableGroup(group={self.group},group_alias={self.group_alias})"
