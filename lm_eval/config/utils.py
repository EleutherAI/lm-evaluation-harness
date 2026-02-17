import ast
from typing import TYPE_CHECKING, Any

from lm_eval import utils


if TYPE_CHECKING:
    from lm_eval.config.task import _MetricConfig


def process_field(
    doc: dict[str, str],
    field_spec: Any | None,
    *,
    digits: bool = False,
    lists: bool = False,
    default: Any | None = None,
) -> Any:
    """Processes a field from a document."""
    # fmt: off
    match field_spec:
        case None: return default
        case func if callable(field_spec): return func(doc)
        case int(): return field_spec
        case list(): return field_spec
        case str() if field_spec in doc: return doc[field_spec]
    # fmt: on

    target_string = utils.apply_template(field_spec, doc)
    if lists:
        # TODO: fix sequence
        if isinstance(target_string, list) and any(
            x in ["{", "}", "(", ")", "[", "]"] for x in target_string
        ):
            return [utils.apply_template(x, doc) for x in target_string]
        return ast.literal_eval(target_string)
    elif digits:
        return int(target_string) if target_string.isdigit() else target_string

    return target_string or default


def parse_metric(cfg: "_MetricConfig"):
    from lm_eval.config.metric import Metric

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
    _higher_is_better = cfg.get("higher_is_better", metric.higher_is_better) or True

    return Metric(
        name=_metric_name,
        fn=metric.fn,
        aggregation=_agg_fn,
        reduction=_reduce_fn,
        higher_is_better=_higher_is_better,
        kwargs=cfg.get("kwargs") or {},
    )
