from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, cast

from typing_extensions import Self, TypeVar

from lm_eval.config.utils import parse_metric


if TYPE_CHECKING:
    from lm_eval.config.task import _MetricConfig


_T = TypeVar("_T")

METRIC_KEYS = {"metric", "aggregation", "higher_is_better", "kwargs"}


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
