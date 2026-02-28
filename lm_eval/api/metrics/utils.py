from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast


if TYPE_CHECKING:
    from collections.abc import Callable

    from lm_eval.config.task import MetricConfig

    from .metric import Metric


eval_logger = logging.getLogger(__name__)


def _metric_with_defaults(name: str, kwargs: dict[str, Any]):
    from dataclasses import replace

    from lm_eval.api.registry import metric_registry

    metric = metric_registry.get(name, None)
    if metric is not None:
        return replace(metric, kwargs=kwargs or {**metric.kwargs})


def _resolve_registry_fn(value, lookup_fn, label: str) -> Callable | None:
    """Resolve a string name via *lookup_fn*, pass through callables, or return None."""
    if value is None:
        return None
    if callable(value):
        return value
    if isinstance(value, str):
        fn = lookup_fn(value)
        if fn is None:
            raise ValueError(
                f"{label} '{value}' not found in registry. "
                f"Please ensure it is registered or provide a callable function directly."
            )
        return fn
    return None


def parse_metric(cfg: MetricConfig, output_type: str | None = None) -> Metric[Any, Any]:
    from lm_eval.api.registry import get_aggregation, get_reduction, metric_registry

    from .metric import Metric

    if "metric" not in cfg:
        raise ValueError(
            f"MetricConfig requires a 'metric' field, either a string reference "
            f"to a registered metric or a callable function. Received {cfg}"
        )

    raw = cfg["metric"]
    _output_type = output_type or "generate_until"

    # 1) Resolve the base metric from registry or callable
    if isinstance(raw, str):
        # in the lambda case as we allow arbitrary metrics to be returned from process_results
        base = metric_registry.get(raw, None)
        if base is None:
            eval_logger.warning(
                "Metric '%s' not found in registry. Using a placeholder that "
                "expects values from 'process_results'.",
                raw,
            )
            base = Metric(
                name=raw, fn=lambda *args, **kwargs: -1.0, output_type={_output_type}
            )
        if output_type and output_type not in base.output_type:
            raise ValueError(
                f"metric {base.name} is defined but has output_type '{base.output_type}' which does not match expected output_type(s) '{output_type}'."
            )
    elif callable(raw):
        name = getattr(raw, "__name__", "metric(undefined)")
        base = Metric(name=name, fn=raw, output_type={_output_type})
    else:
        raise TypeError(
            f"'metric' must be a string or callable, got {type(raw)} in {cfg}"
        )

    # 2) Apply cfg overrides, falling back to base metric defaults
    aggregation = _resolve_registry_fn(
        cfg.get("aggregation", base.aggregation), get_aggregation, "Aggregation"
    )
    reduction = _resolve_registry_fn(
        cfg.get("reduction", base.reduction), get_reduction, "Reduction"
    )
    higher_is_better = cfg.get("higher_is_better", base.higher_is_better)
    if aggregation is None:
        eval_logger.warning(
            "Metric '%s' is defined but has no aggregation. Using default aggregation 'mean'.",
            base.name,
        )
        aggregation = cast("Callable[list[float], float]", get_aggregation("mean"))
    if higher_is_better is None:
        eval_logger.debug(
            "Metric '%s' does not specify 'higher_is_better'. Defaulting to True.",
            base.name,
        )
        higher_is_better = True

    return Metric(
        name=base.name,
        fn=base.fn,
        aggregation=aggregation,
        reduction=reduction,
        higher_is_better=higher_is_better,
        kwargs=cfg.get("kwargs") or {},
        output_type={_output_type},
    )
