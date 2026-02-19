import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, cast

from typing_extensions import Protocol, Self, TypeVar

from lm_eval.config.utils import parse_metric


if TYPE_CHECKING:
    from lm_eval.config.task import _MetricConfig


_T = TypeVar("_T")
_K = TypeVar("_K")


class MetricFn(Protocol[_T]):
    """Callable that computes a per-sample metric value."""

    def __call__(self, references: Any, predictions: Any, **kwargs: Any) -> _T: ...


class ReductionFn(Protocol[_T, _K]):
    """Callable that reduces per-repeat scores into one value per document."""

    def __call__(self, references: Any, predictions: Sequence[_T]) -> _K: ...


class AggregationFn(Protocol[_K]):
    """Callable that aggregates per-document values into a corpus-level float."""

    def __call__(self, values: Sequence[_K]) -> float: ...


METRIC_KEYS = {"metric", "aggregation", "higher_is_better", "reduction", "kwargs"}


def filter_kwargs(fn, kwargs) -> Mapping[str, Any]:
    params = inspect.signature(fn).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs  # function accepts **kwargs, pass everything
    return {
        k: v
        for k, v in kwargs.items()
        if k in params
        and params[k].kind
        not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
        )
    }


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
            raise ValueError(
                f"Metric '{self.name}' fn must be callable, got {type(self.fn)}."
            )
        if self.aggregation is not None and not callable(self.aggregation):
            raise ValueError(
                f"Metric '{self.name}' aggregation must be callable, got {type(self.aggregation)}."
            )
        if self.reduction is None:
            object.__setattr__(self, "reduction", take_first)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Self:
        _cfg = {k: v for k, v in cfg.items() if k in METRIC_KEYS}
        if len(_cfg) < len(cfg):
            _cfg["kwargs"] = {k: v for k, v in cfg.items() if k not in METRIC_KEYS}
        return parse_metric(cast("_MetricConfig", _cfg))

    def compute(self, *args: Any, **kwargs: Any) -> _T:
        """Compute the metric for a sample."""
        return self.fn(*args, **filter_kwargs(self.fn, {**self.kwargs, **kwargs}))

    def aggregate(self, values: Sequence[_K]) -> float:
        """Aggregate a list of metric values into a single score."""
        if self.aggregation is None:
            raise ValueError(
                f"Metric {self.name} does not have an aggregation function."
            )
        return self.aggregation(values)
