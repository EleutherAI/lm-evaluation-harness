import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from lm_eval.api.registry import (
    get_metric,
    higher_is_better_registry,
    metric_agg_registry,
    metric_registry,
)


eval_logger = logging.getLogger(__name__)


@dataclass
class MetricConfig:
    """Encapsulates information about a single metric.

    This is the canonical representation for metrics used throughout lm_eval,
    both in the registry and when parsing from YAML configs.
    """

    name: str = ""
    fn: Callable | None = None
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    aggregation_fn: Callable[..., float | int] = lambda x: x
    is_bypass: bool = False
    higher_is_better: bool = True
    hf_evaluate: bool = False
    output_type: str = "generate_until"
    repeat_reduction: list[str] | None = None

    @classmethod
    def from_yaml_field(cls, cfg: Mapping[str, Any], task: str = ""):
        _metric_fn = None
        _metric_name = cfg["metric"]
        hf_evaluate = cfg.get("hf_evaluate", False)
        if callable(_metric_name):
            _metric_name = _metric_name.__name__
        elif isinstance(_metric_name, str):
            _metric_fn = get_metric(_metric_name, hf_evaluate)

        _agg_name = cfg.get("aggregation", None)
        if _agg_name is not None:
            if callable(_agg_name):  # noqa: E721
                _aggregation = cfg["aggregation"]
            else:
                _aggregation = metric_agg_registry.get(_agg_name)
        else:
            # check if metric has an aggregation defined
            _inverse_agg = metric_registry.get(_metric_name, None)
            if _inverse_agg is not None:
                _aggregation = _inverse_agg.aggregation_fn
                _inverse_agg = _aggregation.__name__
            else:
                _inverse_agg = "mean"
                _aggregation = metric_agg_registry.get("mean")

            # inverse_agg = {v: k for k, v in metric_agg_registry.items()}
            # _aggregation = metric_agg_registry.get(_metric_name)
            eval_logger.warning(
                f"[Task: {task}] metric {_metric_name} is defined, but aggregation is not. "
                f"Using default "
                f"aggregation={_inverse_agg}"
            )

        _higher_is_better = cfg.get("higher_is_better", None)
        if _higher_is_better is None:
            eval_logger.warning(
                f"[Task: {task}] metric {_metric_name} is defined, but higher_is_better is not. "
                f"Using default "
                f"higher_is_better={higher_is_better_registry.get(_metric_name)}"
            )
            _higher_is_better = higher_is_better_registry.get(_metric_name)

        return cls(
            name=_metric_name,
            fn=_metric_fn,
            kwargs=cfg.get("kwargs", {}),
            aggregation_fn=_aggregation,
            higher_is_better=_higher_is_better,
            hf_evaluate=hf_evaluate,
            output_type=cfg.get("output_type", "generate_until"),
            repeat_reduction=cfg.get("requires", None),
        )

    # Backward compatibility aliases
    @property
    def compute(self) -> Callable | None:
        """Alias for fn to maintain backward compatibility with MetricSpec."""
        return self.fn

    @compute.setter
    def compute(self, value: Callable) -> None:
        """Setter for compute to maintain backward compatibility."""
        self.fn = value

    @property
    def aggregate(self) -> Callable | None:
        """Alias for aggregation_fn to maintain backward compatibility with MetricSpec."""
        return self.aggregation_fn

    @aggregate.setter
    def aggregate(self, value: Callable) -> None:
        """Setter for aggregate to maintain backward compatibility."""
        self.aggregation_fn = value

    @cached_property
    def metric_name(self) -> str:
        return self.name

    @cached_property
    def aggregation(self) -> Callable[..., Any] | None:
        from lm_eval.api.registry import get_aggregation

        if self.aggregation_fn is None:
            try:
                return get_aggregation(self.name)
            except (KeyError, ImportError):
                return None
        return self.aggregation_fn

    @cached_property
    def _higher_is_better(self) -> bool | None:
        from lm_eval.api.registry import is_higher_better

        if self.higher_is_better is None:
            try:
                return is_higher_better(self.name)
            except (KeyError, ImportError):
                return None
        return self.higher_is_better

    def compute_metric(self, *args, **kwargs) -> Any:
        """Calculates the metric using the provided function and arguments."""
        if self.fn is None:
            raise ValueError(f"Metric function for {self.name} is not defined.")
        return self.fn(*args, **{**(self.kwargs or {}), **kwargs})

    def compute_aggregation(self, *args, **kwargs) -> Any:
        """Computes the aggregation of the metric values."""
        if self.aggregation_fn is None:
            raise ValueError(f"Aggregation function for {self.name} is not defined.")
        return self.aggregation_fn(*args, **kwargs)
