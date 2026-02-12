from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from lm_eval.api.registry import get_aggregation
from lm_eval.config.metric import Metric


if TYPE_CHECKING:
    from collections.abc import Callable

    from lm_eval.config.task import _MetricConfig


# Keys that are part of _MetricConfig itself (not metric kwargs)
_METRIC_CONFIG_KEYS = frozenset(
    {"metric", "aggregation", "higher_is_better", "reduction", "kwargs"}
)


def parse_metric(cfg: _MetricConfig) -> Metric:
    """Parse a metric configuration dictionary into a Metric object."""
    from lm_eval.api.registry import _get_metric

    if "metric" not in cfg:
        raise ValueError(
            f"MetricConfig requires a 'metric' field, either a string reference "
            f"to a registered metric or a callable function. Received {cfg}"
        )

    # Collect flat kwargs: any top-level keys not in the config schema are
    # treated as metric kwargs (e.g. ignore_case, regexes_to_ignore).
    flat_kwargs: dict[str, Any] = {
        k: v for k, v in cfg.items() if k not in _METRIC_CONFIG_KEYS
    }
    # Merge with explicit kwargs dict (explicit kwargs take precedence)
    extra_kwargs: dict[str, Any] = {**flat_kwargs, **cfg.get("kwargs", {})}

    # Get metric. If str, look up registry; if callable, use directly
    _metric_fn = cfg["metric"]

    if isinstance(_metric_fn, str):
        _name = _metric_fn
        _metric = _get_metric(_metric_fn)
        assert _metric is not None, (
            f"Metric '{_metric_fn}' not found in registry. "
            f"Please ensure it is registered or provide a callable function directly."
        )
    else:
        assert callable(_metric_fn)
        _name: str = (
            _metric_fn.__name__ if hasattr(_metric_fn, "__name__") else str(_metric_fn)
        )  # ty:ignore[invalid-assignment]
        _metric = Metric(name=_name, fn=cast("Callable", _metric_fn))

    # look up aggregation
    _agg = cfg.get("aggregation", _metric.aggregation)
    if isinstance(_agg, str):
        agg_fn = get_aggregation(_agg)
        assert agg_fn is not None, (
            f"Aggregation metric '{_agg}' not found in registry. "
            f"Please ensure it is registered or provide a callable function directly."
        )
    elif callable(_agg):
        agg_fn = cast("Callable", _agg)
    else:
        agg_fn = _metric.aggregation
    assert callable(agg_fn), (
        f"Aggregation function must be callable. Got {agg_fn} of type {type(agg_fn)}"
    )

    # look up reduction
    _reduce = cfg.get("reduction", _metric.reduce_fn)
    if isinstance(_reduce, str):
        reduce_fn = get_aggregation(_reduce)
        assert reduce_fn is not None, (
            f"Reduction metric '{_reduce}' not found in registry. "
            f"Please ensure it is registered or provide a callable function directly."
        )
    elif callable(_reduce):
        reduce_fn = cast("Callable", _reduce)
    else:
        reduce_fn = _metric.reduce_fn
    if reduce_fn is not None:
        assert callable(reduce_fn), (
            f"Reduction function must be callable. Got {reduce_fn} of type {type(reduce_fn)}"
        )

    # Merge kwargs: start with registry defaults, overlay with config
    merged_kwargs = {**_metric.kwargs, **extra_kwargs}

    return Metric(
        name=_name,
        fn=_metric.fn,
        kwargs=merged_kwargs,
        aggregation=agg_fn,
        higher_is_better=cfg.get("higher_is_better", _metric.higher_is_better),
        output_type=_metric.output_type,
        reduce_fn=reduce_fn,
    )
