from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class AggMetricConfig:
    """Configuration for how to aggregate a metric across a group's children.

    Maps to the entries in ``aggregate_metric_list`` in a group YAML file::

        aggregate_metric_list:
          - metric: acc
            filter_list: ["none"]
            aggregation: mean
            weight_by_size: true

    Attributes:
        metric: Name of the metric to aggregate (e.g. "acc", "acc_norm").
        filter_list: Filter names to aggregate over (e.g. ["none"]).
            If None, filters are auto-discovered from child task results.
            A bare string is normalized to a single-element list.
        aggregation: Aggregation function. Currently only "mean" is supported
            as a built-in; a custom callable may also be passed.
        weight_by_size: If True, weight each subtask's contribution by its
            sample count when aggregating. Defaults to True.
    """

    metric: str
    filter_list: list[str] | None = None
    aggregation: str | Callable = "mean"
    weight_by_size: bool = True

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
    from YAML should be fed through this dataclass so that loose input types
    (single strings, bare dicts, etc.) are normalized into canonical forms
    before the rest of the system sees them.

    Example YAML::

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

    Attributes:
        group: Unique identifier for the group (e.g. "mmlu").
        group_alias: Optional display name shown in output tables.
            Defaults to ``group`` when not set.
        task: Child task/group references. Can be a single name, a list of
            names, or a list of dicts.
        aggregate_metric_list: Metrics to aggregate across children.
            Accepts a single AggMetricConfig, a dict, or a list of either;
            ``__post_init__`` normalizes everything to ``list[AggMetricConfig]``.
        metadata: Arbitrary user-defined metadata (e.g. version, description).
    """

    group: str
    group_alias: str | None = None
    task: str | list[str | dict[str, str | dict[str, str]]] | None = None
    # Accepts loose YAML input; __post_init__ normalizes to list[AggMetricConfig] | None
    aggregate_metric_list: list[AggMetricConfig] | list[dict] | None = None
    metadata: dict[str, Any] | None = None

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
        from dataclasses import asdict

        cfg_dict = asdict(self)
        for k, v in list(cfg_dict.items()):
            if callable(v):
                cfg_dict[k] = self.serialize_function(v, keep_callable=keep_callable)
        return cfg_dict

    def serialize_function(
        self, value: Callable | str, keep_callable=False
    ) -> Callable | str:
        from inspect import getsource

        if keep_callable:
            return value
        try:
            return getsource(value)  # type: ignore
        except (TypeError, OSError):
            return str(value)
