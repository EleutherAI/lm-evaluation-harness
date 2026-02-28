from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from functools import cache
from typing import TYPE_CHECKING, Any, Generic
from typing_extensions import TypeVar


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lm_eval.config.task import MetricConfig

    from ._types import AggregationFn, MetricFn, ReductionFn


_T = TypeVar("_T", default=float)
_K = TypeVar("_K", default=float)

METRIC_KEYS = {"metric", "aggregation", "higher_is_better", "reduction", "kwargs"}


@cache
def _get_eligible_params(fn) -> frozenset[str] | None:
    """Return the set of keyword-eligible parameter names for *fn*, cached.

    Returns ``None`` as a sentinel when the function accepts ``**kwargs``
    (meaning all keyword arguments should be forwarded).
    """
    params = inspect.signature(fn).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return None
    return frozenset(
        k
        for k, p in params.items()
        if p.kind
        not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.VAR_POSITIONAL)
    )


def filter_kwargs(fn, kwargs) -> Mapping[str, Any]:
    eligible = _get_eligible_params(fn)
    if eligible is None:
        return kwargs  # function accepts **kwargs, pass everything
    return {k: v for k, v in kwargs.items() if k in eligible}


def take_first(references: Any, predictions: Sequence[Any]):
    return predictions[0] if isinstance(predictions, list) else predictions


@dataclass(frozen=True)
class Metric(Generic[_T, _K]):
    """Encapsulates information about a single metric.

    This is the canonical representation for metrics used throughout lm_eval.

    Type Parameters:
        _T: Per-sample result type from ``fn``.
        _K: Reduced type after collapsing repeats via ``reduction``.

    Type chain: ``fn(...) -> _T``, ``reduction(...) -> _K``, ``aggregation(Sequence[_K]) -> float``.
    """

    name: str
    fn: MetricFn[_T]
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    aggregation: AggregationFn[_K] | None = None
    higher_is_better: bool = True
    output_type: str = "multiple_choice"
    reduction: ReductionFn[_T, _K] | None = take_first

    def __post_init__(self):
        if not self.name:
            raise ValueError("Metric name must be non-empty.")
        if not callable(self.fn):
            raise TypeError(
                f"Metric '{self.name}' fn must be callable, got {type(self.fn)}."
            )
        if self.aggregation is not None and not callable(self.aggregation):
            raise ValueError(
                f"Metric '{self.name}' aggregation must be callable, got {type(self.aggregation)}."
            )
        if self.reduction is None:
            object.__setattr__(self, "reduction", take_first)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | MetricConfig) -> Metric[Any, Any]:
        from lm_eval.api._metrics import utils
        from lm_eval.config.utils import normalize_metric_cfg

        return utils.parse_metric(normalize_metric_cfg(cfg))

    def compute(self, *args: Any, **kwargs: Any) -> _T | dict[str, list[_T]]:
        """Compute the metric for a sample."""
        return self.fn(*args, **filter_kwargs(self.fn, {**self.kwargs, **kwargs}))

    def aggregate(self, values: Sequence[_K]) -> float:
        """Aggregate a list of metric values into a single score."""
        if self.aggregation is None:
            raise ValueError(
                f"Metric {self.name} does not have an aggregation function."
            )
        return self.aggregation(values)
