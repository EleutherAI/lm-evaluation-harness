from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class AggMetricConfig:
    """Configuration for how to aggregate a metric across a group's children.

    Maps to the entries in ``aggregate_metric_list`` in a group YAML file.

    Example:
        ```yaml
        aggregate_metric_list:
          - metric: acc
            filter_list: ["none"]
            aggregation: mean
            weight_by_size: true
        ```
    """

    metric: str
    """Name of the metric to aggregate across subtasks (e.g. ``"acc"``,
    ``"exact_match"``). All children must report a metric with this name."""

    filter_list: list[str] | None = None
    """Filter pipeline names to aggregate over (e.g. ``["none"]``,
    ``["strict-match"]``). If None, filters are auto-discovered from
    child task results. A bare string is normalized to a single-element list."""

    aggregation: str | Callable = "mean"
    """Aggregation function to combine per-subtask metrics. Currently only
    ``"mean"`` is supported as a built-in; a custom callable may also be
    passed."""

    weight_by_size: bool = True
    """If True (default), micro-average: weight each subtask's metric by its
    sample count. If False, macro-average: each subtask contributes equally
    regardless of size."""

    def __post_init__(self):
        if self.aggregation != "mean" and not callable(self.aggregation):
            raise ValueError(
                f"Currently, 'mean' is the only pre-defined aggregation. Got '{self.aggregation}'."
            )
        # Handle filter_list: None means auto-discover, string becomes list
        if self.filter_list is None:
            pass  # Keep as None for auto-discovery
        elif isinstance(self.filter_list, str):
            self.filter_list = [self.filter_list]


@dataclass
class GroupConfig:
    """Typed representation of a group YAML configuration.

    This is the ground-truth schema for group YAML files. Raw dicts parsed
    from YAML are fed through this dataclass so that loose input types
    (single strings, bare dicts, etc.) are normalized into canonical forms.

    Example:
        ```yaml
        group: mmlu
        group_alias: MMLU
        task:
          - mmlu_anatomy
          - mmlu_biology
          - group: mmlu_chemistry
            task:
              - mmlu_elementary_chemistry
          - task: some_other_task
        aggregate_metric_list:
          - metric: acc
            filter_list: ["none"]
            aggregation: mean
            weight_by_size: true
        metadata:
          version: 1.0
        ```
    """

    group: str
    """Unique identifier for the group, used for CLI selection
    (e.g. ``--tasks mmlu``)."""

    group_alias: str | None = None
    """Optional display name shown in result tables instead of ``group``."""

    task: str | list[str | dict[str, str | dict[str, str]]] | None = None
    """Child task and/or group references. Can be a single name, a list of
    names, or a list of dicts for inline overrides and nested groups.
    A bare string is normalized to a single-element list."""

    include: str | dict[str, Any] | None = None
    """Task-level defaults applied to every child in this group.

    Can be a path (str) to a YAML file with task fields, or an inline dict
    of key-value pairs. When a path is given it is resolved relative to the
    group YAML file's directory.

    Example (path):
        ```yaml
        group: my_bench
        include: shared_defaults.yaml
        task:
          - task_a
          - task_b
        ```

    Example (inline):
        ```yaml
        group: my_bench
        include:
          num_fewshot: 5
          doc_to_text: "{{question}}"
        task:
          - task_a
          - task_b
        ```
    """

    # Accepts loose YAML input; __post_init__ normalizes to list[AggMetricConfig] | None
    aggregate_metric_list: list[AggMetricConfig] | list[dict] | None = None
    """Metrics to aggregate across child tasks. Without this, the group
    appears as a header row with no aggregate score. Accepts a single
    ``AggMetricConfig``, a dict, or a list of either."""

    metadata: dict[str, Any] | None = None
    """Arbitrary metadata stored alongside results (e.g. ``{"version": 1.0}``).
    The ``num_fewshot`` key overrides the displayed n-shot column for the
    group in result tables."""

    def __post_init__(self):
        if isinstance(self.task, str):
            self.task = [self.task]
        if self.aggregate_metric_list is not None:
            if isinstance(self.aggregate_metric_list, (dict, AggMetricConfig)):
                self.aggregate_metric_list = [self.aggregate_metric_list]
            self.aggregate_metric_list = [
                AggMetricConfig(**item) if isinstance(item, dict) else item  # type:ignore[invalid-argument-type]
                for item in self.aggregate_metric_list
            ]

    def to_dict(self, keep_callable: bool = False) -> dict[str, str]:
        from lm_eval.config.utils import serialize_config

        return serialize_config(self, keep_callable=keep_callable)
