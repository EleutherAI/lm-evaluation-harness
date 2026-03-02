"""Config normalization and serialization utilities.

Functions in this module run at **parse time** to canonicalise raw YAML
dicts into well-typed config structures (e.g. separating known metric
fields from extra kwargs).  They are intentionally separated from the
runtime template helpers in ``config/templates.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast


if TYPE_CHECKING:
    from collections.abc import Mapping

    from lm_eval.config.task import FilterPipeline, FilterStep, MetricConfig


# ── Serialization ────────────────────────────────────────────────────


def serialize_callable(
    value: Callable[..., Any], *, keep_callable: bool = False
) -> Callable[..., Any] | str:
    """Serialize a callable to its source code string.

    If *keep_callable* is True the original callable is returned unchanged.
    Otherwise, we attempt ``inspect.getsource``; on failure we fall back to ``str()``.
    """
    from inspect import getsource

    if keep_callable:
        return value
    try:
        return getsource(value)
    except (TypeError, OSError):
        return str(value)


def _serialize_value(value: Any, keep_callable: bool) -> Any:
    """Recursively serialize callables in nested dicts/lists."""
    if callable(value):
        return serialize_callable(value, keep_callable=keep_callable)
    if isinstance(value, dict):
        return {k: _serialize_value(v, keep_callable) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize_value(item, keep_callable) for item in value]
    return value


def serialize_config(
    cfg, *, keep_callable: bool = False
) -> dict[str, str] | dict[str, Any]:
    """Convert a dataclass config to a plain dict, serializing callables.

    * ``None`` values are dropped.
    * Any callable value is serialized with [serialize_callable][serialize_callable].
    """
    from dataclasses import asdict

    cfg_dict = asdict(cfg)
    for k, v in list(cfg_dict.items()):
        if v is None:
            cfg_dict.pop(k)
        else:
            cfg_dict[k] = _serialize_value(v, keep_callable)
    return cfg_dict


# ── Metric normalization ─────────────────────────────────────────────


def _norm_kwargs(cfg: Mapping[str, Any], keys: set[str]) -> dict[str, Any]:
    return {k: v for k, v in cfg.items() if k in keys} | {
        "kwargs": {k: v for k, v in cfg.items() if k not in keys and k != "kwargs"}
        | cfg.get("kwargs", {})
    }


def normalize_metric_cfg(cfg: Mapping[str, Any]) -> MetricConfig:
    """Normalize a raw YAML metric entry into a proper MetricConfig.

    YAML metric entries mix known fields (``metric``, ``aggregation``, etc.)
    with arbitrary extra keys that should be forwarded as ``kwargs`` to the
    metric function. This function separates the two.

    Examples:
        >>> normalize_metric_cfg({"metric": "exact_match", "ignore_case": True})
        {"metric": "exact_match", "kwargs": {"ignore_case": True}}
    """
    METRIC_KEYS = {"metric", "aggregation", "higher_is_better", "reduction", "kwargs"}

    if "metric" not in cfg:
        raise ValueError(
            f"Each metric_list entry requires a 'metric' field. Got: {dict(cfg)}"
        )
    if not isinstance(_m := (cfg["metric"]), (str, Callable)):
        raise TypeError(
            f"'metric' must be a string or callable, got {type(_m)} in {cfg}"
        )

    return cast("MetricConfig", _norm_kwargs(cfg, METRIC_KEYS))


def normalize_metric_list(
    cfg: list[Mapping[str, Any]] | list[MetricConfig],
) -> list[MetricConfig]:
    """Normalize a raw ``metric_list`` from YAML into a list of MetricConfigs.

    Returns an empty list when *cfg* is empty — the scorer layer is
    responsible for providing defaults (via ``Scorer.default_metric_cfg``
    or ``DEFAULT_METRIC_REGISTRY``).
    """
    if not cfg:
        return []
    return [normalize_metric_cfg(entry) for entry in cfg]


# ── Filter normalization ─────────────────────────────────────────────

FILTER_STEP_KEYS = {"function", "kwargs"}


def _normalize_filter_step(cfg: Mapping[str, Any]) -> FilterStep:
    r"""Normalize a raw YAML filter step into a proper FilterStep.

    Same pattern as [normalize_metric_cfg][normalize_metric_cfg]: known fields are kept,
    extra keys (e.g. ``regex_pattern``) are collected into ``kwargs``.

    Examples:
        >>> _normalize_filter_step({"function": "regex", "regex_pattern": r"\\d+"})
        {"function": "regex", "kwargs": {"regex_pattern": "\\\\d+"}}
    """
    if "function" not in cfg:
        raise KeyError(f"Each filter step requires a 'function' field. Got: {cfg}")
    return cast("FilterStep", _norm_kwargs(cfg, FILTER_STEP_KEYS))


def normalize_filter_list(
    cfg: list[Mapping[str, Any]] | list[FilterPipeline],
) -> list[FilterPipeline]:
    """Normalize a raw ``filter_list`` from YAML into a list of FilterPipelines.

    Each pipeline's ``filter`` steps and nested ``metric_list`` are normalized.
    When *cfg* is empty, returns an empty list (the caller provides a default scorer).
    """
    if not cfg:
        return []

    result = []
    for pipeline in cfg:
        if "name" not in pipeline or "filter" not in pipeline:
            raise KeyError(
                f"'name' and 'filter' are required keys for each filter pipeline, got {list(pipeline.keys())} in {cfg}"
            )
        entry: FilterPipeline = {
            "name": pipeline["name"],
            "filter": [_normalize_filter_step(s) for s in pipeline["filter"]],
        }

        if "metric_list" in pipeline:
            entry["metric_list"] = normalize_metric_list(pipeline["metric_list"])
        result.append(entry)
    return result
