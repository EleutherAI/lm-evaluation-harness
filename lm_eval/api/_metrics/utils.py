from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from lm_eval.config.task import _MetricConfig


def parse_metric(cfg: _MetricConfig):
    from lm_eval.api._metrics.metric import Metric

    if "metric" not in cfg:
        raise ValueError(
            f"MetricConfig requires a 'metric' field, either a string reference "
            f"to a registered metric or a callable function. Received {cfg}"
        )
    _metric = cfg["metric"]

    # look up metric key
    if isinstance(_metric, str):
        from lm_eval.api.registry import _get_metric

        _metric_name = _metric
        metric = _get_metric(_metric)
        # use defaults from metric if not specified in cfg
        if metric is not None and "aggregation" not in cfg and "reduction" not in cfg:
            return Metric(
                name=_metric_name,
                fn=metric.fn,
                aggregation=metric.aggregation,
                reduction=metric.reduction,
                kwargs=cfg.get("kwargs") or {},
            )

        if metric is None:
            # We allow metrics not in registry (e.g if a user overloads process_results). Create a dummy metric
            metric = Metric(name=_metric_name, fn=lambda *args, **kwargs: None)

    elif callable(_metric):
        _metric_name = getattr(_metric, "__name__", "operation")
        metric = Metric(name=getattr(_metric, "__name__", "operation"), fn=_metric)

    # look up aggregations
    _agg = cfg.get("aggregation", metric.aggregation)
    _agg_fn = None
    if isinstance(_agg, str):
        from lm_eval.api.registry import get_aggregation

        _agg_fn = get_aggregation(_agg)
        if _agg_fn is None:
            raise ValueError(
                f"Aggregation metric '{_agg}' not found in registry. "
                f"Please ensure it is registered or provide a callable function directly."
            )
    elif callable(_agg):
        _agg_fn = _agg

    # look up reductions
    _reduce = cfg.get("reduction", metric.reduction)
    _reduce_fn = None
    if isinstance(_reduce, str):
        from lm_eval.api.registry import get_reduction

        _reduce_fn = get_reduction(_reduce)
        if _reduce_fn is None:
            raise ValueError(
                f"Reduction metric '{_reduce}' not found in registry. "
                f"Please ensure it is registered or provide a callable function directly."
            )
    elif callable(_reduce):
        _reduce_fn = _reduce

    # higher_is_better semantics
    _hib = cfg.get("higher_is_better", metric.higher_is_better)
    _higher_is_better = _hib if _hib is not None else True

    return Metric(
        name=_metric_name,
        fn=metric.fn,
        aggregation=_agg_fn,
        reduction=_reduce_fn,
        higher_is_better=_higher_is_better,
        kwargs=cfg.get("kwargs") or {},
    )


def softmax(x: ArrayLike) -> np.ndarray:
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
