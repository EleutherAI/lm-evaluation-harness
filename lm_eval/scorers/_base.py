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


@dataclass(frozen=True, slots=True)
class ScoredDoc:
    """Per-document scoring result produced by a Scorer.

    Bundles the document reference and all metric scores together so that
    downstream reduction never needs to align parallel lists.
    """

    doc_id: int
    reference: Any  # str for gen, int|list[int] for MC
    scores: dict[str, list[float]]  # {metric_name: [per_repeat_values]}


@dataclass(frozen=True, slots=True)
class MetricKey:
    """Structured representation of a ``"metric,scorer"`` key."""

    metric: str
    scorer: str
    is_stderr: bool = False

    def __str__(self) -> str:
        name = f"{self.metric}_stderr" if self.is_stderr else self.metric
        return f"{name},{self.scorer}"

    @classmethod
    def parse(cls, key: str) -> MetricKey | None:
        """Parse a ``'metric,scorer'`` string. Returns ``None`` if not a metric key."""
        if "," not in key:
            return None
        left, _, scorer = key.partition(",")
        if left.endswith("_stderr"):
            return cls(metric=left[: -len("_stderr")], scorer=scorer, is_stderr=True)
        return cls(metric=left, scorer=scorer)


@dataclass
class Scorer:
    name: str
    filter: FilterEnsemble
    metrics: list[Metric] | None = None
    output_type: str | None = None
    _scored_docs: list[ScoredDoc] = field(default_factory=list)
    _reduced_results: dict[str, list[Any]] = field(default_factory=dict)

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

    def score_instances(self, instances: dict[int, list[Instance]]) -> list[ScoredDoc]:
        """Score all documents' instances, returning a ``ScoredDoc`` per document.

        For each doc_id group, builds a structured result object
        (``LLResults`` or ``GenResults``) from the instances and dispatches
        metrics against it.

        For ``generate_until`` tasks, each repeat is scored independently,
        producing ``list[T]`` per doc per metric.  For other output types
        (repeats always 1), the scalar is wrapped in a single-element list
        so that the downstream ``reduce`` step works uniformly.
        """
        from lm_eval._types import LLResults

        scored_docs: list[ScoredDoc] = []

        for doc_id, doc_instances in instances.items():
            if self.output_type == "generate_until":
                # Per-repeat scoring for generate_until
                inst = doc_instances[0]  # 1 instance per doc for generate_until
                resps = inst.filtered_resps[self.name]  # [str * K]
                target = inst.target

                repeat_scores: dict[str, list] = defaultdict(list)
                for resp in resps:
                    per_repeat = self._dispatch_metrics([target], [resp])
                    for metric_name, value in per_repeat.items():
                        repeat_scores[metric_name].append(value)

                scored_docs.append(
                    ScoredDoc(
                        doc_id=doc_id,
                        reference=target,
                        scores=dict(repeat_scores),
                    )
                )
            else:
                # loglikelihood / multiple_choice — repeats=1
                results_obj = LLResults.from_instances(doc_instances)
                references = results_obj.targets
                per_doc = self._dispatch_metrics(references, results_obj)

                scored_docs.append(
                    ScoredDoc(
                        doc_id=doc_id,
                        reference=references,
                        scores={mn: [v] for mn, v in per_doc.items()},
                    )
                )

        return scored_docs

    def _dispatch_metrics(self, references: Any, predictions: Any) -> dict[str, Any]:
        """Call each Metric.compute(references, predictions) and collect per-doc results."""
        result_dict: dict[str, Any] = {}
        if not self.metrics:
            return result_dict
        for m in self.metrics:
            score = m.compute(references, predictions)
            if isinstance(score, dict):
                result_dict.update(score)
            else:
                result_dict[m.name] = score
        return result_dict

    # ------------------------------------------------------------------
    # Reduction & Aggregation
    # ------------------------------------------------------------------

    def reduce(self, scored_docs: list[ScoredDoc]) -> dict[str, list]:
        """Reduce per-doc list[T] -> T for each document.

        Calls ``m.reduction(reference, predictions)`` where *reference* is
        the raw target for the document and *predictions* is the list of K
        per-repeat scored values.
        """
        reduced: dict[str, list] = {}
        for m in self.metrics or []:
            values = []
            for sd in scored_docs:
                if m.name in sd.scores:
                    values.append(m.reduction(sd.reference, sd.scores[m.name]))
            if values:
                reduced[m.name] = values
        self._reduced_results = reduced
        return reduced

    def aggregate(
        self,
        metric_results: dict[str, list] | None = None,
        bootstrap_iters: int | None = 100000,
    ) -> tuple[dict[str, Any], int]:
        """Aggregate metric results and compute stderr.

        Calls ``m.aggregate(values)`` uniformly for all metrics — both
        plain function metrics and corpus metrics have their aggregation
        captured on the ``Metric`` object at construction time.

        Returns ``(agg_metrics, sample_len)`` where keys are in
        ``"metric,{self.name}"`` / ``"metric_stderr,{self.name}"`` format.
        """
        from lm_eval.api.metrics import mean, stderr_for_metric

        results = (
            metric_results if metric_results is not None else self._reduced_results
        )
        agg: dict[str, Any] = {}
        sample_len = 0

        for m in self.metrics or []:
            if m.name not in results:
                continue
            values = results[m.name]
            sample_len = max(sample_len, len(values))

            # Aggregation — Metric.aggregate() works uniformly for both
            # plain function metrics and corpus metrics (aggregation was
            # captured at construction time in _get_metric).
            agg_fn = m.aggregation
            if agg_fn is not None:
                agg[str(MetricKey(m.name, self.name))] = m.aggregate(values)
            else:
                eval_logger.warning(f"No aggregation for {m.name}. Using mean.")
                agg_fn = mean
                agg[str(MetricKey(m.name, self.name))] = mean(values)

            # Stderr
            stderr_key = str(MetricKey(m.name, self.name, is_stderr=True))
            if isinstance(bootstrap_iters, int) and bootstrap_iters > 0:
                stderr_fn = stderr_for_metric(
                    metric=agg_fn, bootstrap_iters=bootstrap_iters
                )
                agg[stderr_key] = (
                    stderr_fn(values) if (stderr_fn and len(values) > 1) else "N/A"
                )
            else:
                agg[stderr_key] = "N/A"

        return agg, sample_len

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def higher_is_better(self) -> dict[str, bool]:
        """Return ``{metric_name: bool}`` for all metrics in this scorer."""
        return {m.name: m.higher_is_better for m in (self.metrics or [])}
