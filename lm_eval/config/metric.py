import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Generic

from typing_extensions import TypeVar


_T = TypeVar("_T")


def has_kwargs(fn):
    sig = inspect.signature(fn)
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def filter_kwargs(fn, kwargs) -> Mapping[str, Any]:
    params = inspect.signature(fn).parameters
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


@dataclass
class Metric(Generic[_T]):
    """Encapsulates information about a single metric.

    This is the canonical representation for metrics used throughout lm_eval
    """

    name: str
    fn: Callable[..., _T]
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    aggregation: Callable[[list[_T]], float] | None = None
    higher_is_better: bool = True
    output_type: str = "generate_until"
    reduce_fn: Callable[[list[_T]], _T] | None = None

    def compute(self, *args, **kwargs):
        """Compute the metric for a single instance."""
        return (
            self.fn(*args, **{**self.kwargs, **kwargs})
            if has_kwargs(self.fn)
            else self.fn(*args, **filter_kwargs(self.fn, {**self.kwargs, **kwargs}))
        )

    def _compute(self, *args, **kwargs):
        """Alias for compute."""
        return self.compute(*args, **kwargs)

    def reduce(self):
        """Reduce a list of metric results into a single result of the same type."""
        if self.reduce_fn is None:
            raise ValueError(f"Metric {self.name} does not have a reduce function.")
        return self.reduce_fn

    def aggregate(self, results: list[_T]) -> float | int:
        """Aggregate a list of metric results into a single score."""
        if self.aggregation is None:
            raise ValueError(
                f"Metric {self.name} does not have an aggregation function."
            )
        return self.aggregation(results)
