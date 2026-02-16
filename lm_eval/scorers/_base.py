from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from typing_extensions import Self

from lm_eval.api.filter import FilterEnsemble


if TYPE_CHECKING:
    from lm_eval.api.instance import Instance
    from lm_eval.config.metric import Metric

eval_logger = logging.getLogger(__name__)


@dataclass
class Scorer:
    name: str
    filter: FilterEnsemble
    metrics: list[Metric] | None = None
    output_type: str | None = None
    _metric_results: dict[str, list[Any]] = field(
        default_factory=lambda: defaultdict(list)
    )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
        global_metrics: list[Metric] | None = None,
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
                "metric_list": [           # optional -- falls back to global_metrics
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

    @classmethod
    def default_scorer(
        cls, global_metrics: list[Metric], output_type: str | None = None
    ) -> Self:
        """Build the default scorer: ``take_first`` filter with the given metrics."""
        return cls.from_dict(
            {
                "name": "none",
                "filter": [{"function": "take_first"}],
            },
            global_metrics=global_metrics,
            output_type=output_type,
        )

    # ------------------------------------------------------------------
    # Filter
    # ------------------------------------------------------------------

    def apply_filter(self, instances: list[Instance]) -> None:
        self.filter.apply(instances)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear accumulated per-metric results for a fresh scoring pass."""
        self._metric_results = defaultdict(list)

    def score_instances(self, instances: dict[int, list[Instance]]) -> None:
        """Score all documents' instances, accumulating per-metric results.

        For each doc_id group, builds a structured result object
        (``LLResults`` or ``GenResults``) from the instances and dispatches
        metrics against it.
        """
        from lm_eval._types import GenResults, LLResults

        self.reset()

        for doc_id, doc_instances in instances.items():
            # Build structured result object based on output_type
            if self.output_type in (
                "multiple_choice",
                "loglikelihood",
                "loglikelihood_rolling",
            ):
                results_obj = LLResults.from_instances(doc_instances)
            elif self.output_type == "generate_until":
                results_obj = GenResults.from_instances(doc_instances)
            else:
                results_obj = LLResults.from_instances(doc_instances)

            target = results_obj.targets
            per_doc = self._dispatch_metrics(target, results_obj)

            # Accumulate per-doc results into lists keyed by metric name
            for metric_name, value in per_doc.items():
                self._metric_results[metric_name].append(value)

    def _dispatch_metrics(self, targets: Any, results: Any) -> dict[str, Any]:
        """Call each Metric.compute(targets, results) and collect per-doc results."""
        result_dict: dict[str, Any] = {}
        if not self.metrics:
            return result_dict
        for m in self.metrics:
            score = m.compute(targets, results)
            if isinstance(score, dict):
                result_dict.update(score)
            else:
                result_dict[m.name] = score
        return result_dict

    # ------------------------------------------------------------------
    # Reduction & Aggregation
    # ------------------------------------------------------------------

    def _reduce(self) -> dict[str, Any]:
        """Reduce metric results if any Metric has a reduce_fn."""
        reduced_results = {}
        for m in self.metrics or []:
            if m.name in self._metric_results:
                reduced_results[m.name] = m.reduction(self._metric_results[m.name])
        return reduced_results

    def aggregate(self) -> dict[str, float]:
        """Aggregate metric results using each Metric's aggregation function.

        For ``CorpusMetric`` instances, delegates to their ``.aggregation()`` method.
        For regular metrics, calls ``m.aggregate(values)``.
        """
        from lm_eval.api._metrics.corpus import CorpusMetric

        agg_results: dict[str, float] = {}
        for m in self.metrics or []:
            if m.name not in self._metric_results:
                continue
            values = self._metric_results[m.name]
            # CorpusMetric classes have their own aggregation method
            if isinstance(m.fn, CorpusMetric):
                agg_results[m.name] = m.fn.aggregation(values)
            elif m.aggregation is not None:
                agg_results[m.name] = m.aggregate(values)
        return agg_results

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def higher_is_better(self) -> dict[str, bool]:
        """Return ``{metric_name: bool}`` for all metrics in this scorer."""
        return {m.name: m.higher_is_better for m in (self.metrics or [])}
