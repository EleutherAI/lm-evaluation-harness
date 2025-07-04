from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, List, Optional


@dataclass
class MetricConfig:
    """Encapsulates information about a single metric."""

    name: str
    fn: Optional[Callable] = None
    kwargs: Optional[dict] = None
    aggregation_fn: Optional[Callable] = None
    higher_is_better: bool = True
    hf_evaluate: bool = False
    is_elementwise: bool = True

    @cached_property
    def metric_name(self) -> str:
        return self.name

    @cached_property
    def aggregation(self) -> Callable:
        from lm_eval.api.registry import get_aggregation

        if self.aggregation_fn is None:
            return get_aggregation(self.name)
        return self.aggregation_fn

    @cached_property
    def _higher_is_better(self) -> bool:
        from lm_eval.api.registry import is_higher_better

        if self.higher_is_better is None:
            return is_higher_better(self.name)
        return self.higher_is_better

    def compute_metric(self, *args, **kwargs) -> Any:
        """Calculates the metric using the provided function and arguments."""
        if self.fn is None:
            raise ValueError(f"Metric function for {self.name} is not defined.")
        return self.fn(*args, **{**self.kwargs, **kwargs})

    def compute_aggregation(self, values: List[Any]) -> Any:
        """Computes the aggregation of the metric values."""
        if self.aggregation_fn is None:
            raise ValueError(f"Aggregation function for {self.name} is not defined.")
        return self.aggregation_fn(values)
