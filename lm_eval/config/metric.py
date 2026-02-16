import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, cast

from typing_extensions import Self, TypeVar

from lm_eval.config.utils import parse_metric


if TYPE_CHECKING:
    from lm_eval.config.task import _MetricConfig


_T = TypeVar("_T")

METRIC_KEYS = {"metric", "aggregation", "higher_is_better", "kwargs"}


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
    output_type: str = "multiple_choice"
    reduction: Callable[[list[_T]], _T] = lambda x: x[0] if isinstance(x, list) else x

    @classmethod
    def from_dict(self, cfg: dict[str, Any]) -> Self:
        _cfg = {k: v for k, v in cfg.items() if k in METRIC_KEYS}
        if len(_cfg) < len(cfg):
            _cfg["kwargs"] = {k: v for k, v in cfg.items() if k not in METRIC_KEYS}
        return parse_metric(cast("_MetricConfig", _cfg))

    def compute(self, *args, **kwargs):
        """Compute the metric for a sample."""
        return self.fn(*args, **filter_kwargs(self.fn, {**self.kwargs, **kwargs}))

    def aggregate(self, values: list[_T]) -> float:
        """Aggregate a list of metric values into a single score."""
        if self.aggregation is None:
            raise ValueError(
                f"Metric {self.name} does not have an aggregation function."
            )
        return self.aggregation(values)
