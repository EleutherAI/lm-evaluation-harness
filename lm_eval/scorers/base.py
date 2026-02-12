from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable

    from lm_eval.api.filter import FilterEnsemble
    from lm_eval.config.metric import Metric


@dataclass
class Scorer:
    name: str
    filter: FilterEnsemble
    metrics: list[Metric]
    _resps: list = field(default_factory=list)

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
        global_metrics: list[Metric] | None = None,
    ) -> Scorer:
        """Build a Scorer from a filter_list entry dict.

        Expected shape (mirrors the YAML ``filter_list`` entries)::

            {
                "name": "strict-match",
                "filter": [
                    {"function": "take_first"},
                    {"function": "regex", "regex_pattern": "..."},
                ],
                "metric_list": [           # optional – falls back to global_metrics
                    {"metric": "exact_match", "aggregation": "mean", ...},
                ],
            }
        """
        from lm_eval.config.metric import Metric
        from lm_eval.filters import build_filter_ensemble

        global_metrics = global_metrics or []

        # --- build filter ensemble ---
        filter_name = cfg.get("name", "none")
        filter_functions = cfg.get("filter", [{"function": "take_first"}])
        components: list[tuple[str, dict[str, Any] | None]] = []
        for fn_cfg in filter_functions:
            fn_name = fn_cfg["function"]
            kwargs = {k: v for k, v in fn_cfg.items() if k != "function"}
            components.append((fn_name, kwargs or None))
        filter_ensemble = build_filter_ensemble(filter_name, components)

        # --- build metrics ---
        if cfg.get("metric_list"):
            metrics = [Metric.from_yaml(m) for m in cfg["metric_list"]]
        else:
            metrics = list(global_metrics)

        return cls(
            name=filter_name,
            filter=filter_ensemble,
            metrics=metrics,
        )

    @classmethod
    def default_scorer(cls, global_metrics: list[Metric]) -> Scorer:
        """Build the default scorer: ``take_first`` filter with the given metrics."""
        return cls.from_dict(
            {
                "name": "none",
                "filter": [{"function": "take_first"}],
            },
            global_metrics=global_metrics,
        )

    # --- helper properties ---

    @property
    def metric_names(self) -> list[str]:
        return [m.name for m in self.metrics]

    @property
    def aggregation_dict(self) -> dict[str, Callable]:
        """Map metric name -> aggregation function."""
        return {
            m.name: m.aggregation for m in self.metrics if m.aggregation is not None
        }

    @property
    def higher_is_better_dict(self) -> dict[str, bool]:
        """Map metric name -> higher_is_better flag."""
        return {m.name: m.higher_is_better for m in self.metrics}

    # --- instance methods ---

    def apply_filter(self, instances: list) -> None:
        """Apply the filter to the task's instances."""
        self.filter.apply(instances)

    def apply_metrics(self, instances: list) -> dict[str, list]:
        """Apply the scorer's metrics to the task's instances."""
        raise NotImplementedError("Scorer.apply_metrics not implemented yet")
