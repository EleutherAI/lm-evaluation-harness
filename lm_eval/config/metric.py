from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any


@dataclass
class MetricConfig:
    """Encapsulates information about a single metric.

    This is the canonical representation for metrics used throughout lm_eval,
    both in the registry and when parsing from YAML configs.
    """

    name: str
    fn: Callable = lambda x: x
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    aggregation_fn: Callable = lambda x: x
    is_bypass: bool = False
    higher_is_better: bool = True
    hf_evaluate: bool = False
    output_type: str = "generate_until"
    requires: list[str] | None = None

    # Backward compatibility aliases
    @property
    def compute(self) -> Callable | None:
        """Alias for fn to maintain backward compatibility with MetricSpec."""
        return self.fn

    @compute.setter
    def compute(self, value: Callable) -> None:
        """Setter for compute to maintain backward compatibility."""
        self.fn = value

    @property
    def aggregate(self) -> Callable | None:
        """Alias for aggregation_fn to maintain backward compatibility with MetricSpec."""
        return self.aggregation_fn

    @aggregate.setter
    def aggregate(self, value: Callable) -> None:
        """Setter for aggregate to maintain backward compatibility."""
        self.aggregation_fn = value

    @cached_property
    def metric_name(self) -> str:
        return self.name

    @cached_property
    def aggregation(self) -> Callable[..., Any] | None:
        from lm_eval.api.registry import get_aggregation

        if self.aggregation_fn is None:
            try:
                return get_aggregation(self.name)
            except (KeyError, ImportError):
                return None
        return self.aggregation_fn

    @cached_property
    def _higher_is_better(self) -> bool | None:
        from lm_eval.api.registry import is_higher_better

        if self.higher_is_better is None:
            try:
                return is_higher_better(self.name)
            except (KeyError, ImportError):
                return None
        return self.higher_is_better

    def compute_metric(self, *args, **kwargs) -> Any:
        """Calculates the metric using the provided function and arguments."""
        if self.fn is None:
            raise ValueError(f"Metric function for {self.name} is not defined.")
        return self.fn(*args, **{**(self.kwargs or {}), **kwargs})

    def compute_aggregation(self, *args, **kwargs) -> Any:
        """Computes the aggregation of the metric values."""
        if self.aggregation_fn is None:
            raise ValueError(f"Aggregation function for {self.name} is not defined.")
        return self.aggregation_fn(*args, **kwargs)
