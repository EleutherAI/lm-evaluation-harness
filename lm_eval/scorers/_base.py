from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from typing_extensions import Self

from lm_eval.api.filter import FilterEnsemble


if TYPE_CHECKING:
    from lm_eval.api.instance import Instance
    from lm_eval.config.metric import Metric


@dataclass
class Scorer:
    name: str
    filter: FilterEnsemble
    metrics: "list[Metric] | None" = None  # list[Metric]
    output_type: str | None = None
    _resps: list = field(default_factory=list)

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
        global_metrics: list["Metric"] | None = None,
        output_type: str | None = None,
    ) -> Self:
        """Build a Scorer from a filter_list entry dict.

        Expected shape (mirrors the YAML ``filter_list`` entries)::

            {
                "name": "strict-match",
                "filter": [
                    {"function": "take_first"},
                    {"function": "regex", "regex_pattern": "..."},
                ],
                "metric_list": [           # optional – falls back to global_metrics
                    {"metric": "exact_match", "aggregation": "mean", ...},
                ],
            }
        """
        from lm_eval.config.metric import Metric
        from lm_eval.filters import build_filter_ensemble

        global_metrics = global_metrics or []

        # --- build filter ensemble ---
        filter_name = cfg.get("name", "none")
        filter_functions = cfg.get("filter", [{"function": "take_first"}])
        components: list[tuple[str, dict[str, Any] | None]] = []
        for fn_cfg in filter_functions:
            fn_name = fn_cfg["function"]
            kwargs = {k: v for k, v in fn_cfg.items() if k != "function"}
            components.append((fn_name, kwargs or None))
        filter_ensemble = build_filter_ensemble(filter_name, components)

        # --- build metrics ---
        if cfg.get("metric_list"):
            metrics = [Metric.from_dict(m) for m in cfg["metric_list"]]
        else:
            metrics = list(global_metrics)

        return cls(
            name=filter_name,
            filter=filter_ensemble,
            metrics=metrics,
            output_type=output_type,
        )

    def apply_filter(self, instances: list["Instance"]) -> None:
        self.filter.apply(instances)

    def score(self, instances: list["Instance"]):
        raise NotImplementedError
