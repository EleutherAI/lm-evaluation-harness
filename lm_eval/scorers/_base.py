from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from typing_extensions import Self


if TYPE_CHECKING:
    from lm_eval.api.filter import FilterEnsemble


if TYPE_CHECKING:
    from lm_eval.api.instance import Instance
    from lm_eval.config.metric import Metric

eval_logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ScoredDoc:
    """Per-document scoring result produced by a Scorer.

    Bundles the document reference and all metric scores together so that
    downstream reduction never needs to align parallel lists.
    """

    doc_id: int
    reference: Any  # str for gen, int|list[int] for MC, loglikelihood
    scores: dict[str, list[float]]  # {metric_name: [per_repeat_values]}
    reduced_scores: dict[str, float] = field(default_factory=dict)  # post-reduction


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
    _scored_docs: dict[int, ScoredDoc] = field(default_factory=dict)

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

    def score_instances(
        self, instances: dict[int, list[Instance]]
    ) -> dict[int, ScoredDoc]:
        """Score all documents' instances, returning a ``ScoredDoc`` per document.

        For each doc_id group, builds a structured result object
        (``LLResults`` or ``GenResults``) from the instances and dispatches
        metrics against it.

        For ``generate_until`` tasks, each repeat is scored independently,
        producing ``list[T]`` per doc per metric.  For other output types
        (repeats always 1), the scalar is wrapped in a single-element list
        so that the downstream ``reduce`` step works uniformly.
        """
        from lm_eval.api._metrics.results import LLResults

        scored_docs: dict[int, ScoredDoc] = {}

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

                scored_docs[doc_id] = ScoredDoc(
                    doc_id=doc_id,
                    reference=target,
                    scores=dict(repeat_scores),
                )
            elif self.output_type == "loglikelihood_rolling":
                # Rolling loglikelihood: 1 instance per doc, model returns a plain float
                import numpy as np

                inst = doc_instances[0]
                ll = inst.resps[0]  # plain float from loglikelihood_rolling
                text = inst.args[0]  # the scored text (for word/byte counting)
                results_obj = LLResults(
                    results=inst.resps,
                    lls=np.array([ll]),
                    is_greedy=[False],
                    targets=inst.target,
                    ctx="",
                    choices=[text],
                )
                references = results_obj.targets
                per_doc = self._dispatch_metrics(references, results_obj)

                scored_docs[doc_id] = ScoredDoc(
                    doc_id=doc_id,
                    reference=references,
                    scores={mn: [v] for mn, v in per_doc.items()},
                )
            else:
                # loglikelihood / multiple_choice — repeats=1
                results_obj = LLResults.from_instances(doc_instances)
                references = results_obj.targets
                per_doc = self._dispatch_metrics(references, results_obj)

                scored_docs[doc_id] = ScoredDoc(
                    doc_id=doc_id,
                    reference=references,
                    scores={mn: [v] for mn, v in per_doc.items()},
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

    def reduce(self, scored_docs: dict[int, ScoredDoc]) -> None:
        """Reduce per-doc list[T] -> T for each document.

        Iterates over each scored doc's metrics and applies the matching
        ``Metric.reduction`` if available, otherwise takes the first value.

        Writes the reduced scalar onto each ``ScoredDoc.reduced_scores``.
        """
        metrics_by_name = {m.name: m for m in self.metrics or []}
        for sd in scored_docs.values():
            for metric_name, values in sd.scores.items():
                m = metrics_by_name.get(metric_name)
                if m is not None and m.reduction is not None:
                    sd.reduced_scores[metric_name] = m.reduction(sd.reference, values)
                else:
                    # Unknown metric (e.g. from process_results): take first
                    sd.reduced_scores[metric_name] = values[0]

    def aggregate(
        self,
        metric_results: dict[str, list] | None = None,
        bootstrap_iters: int | None = 100000,
    ) -> tuple[dict[str, Any], int]:
        """Aggregate metric results and compute stderr.

        Iterates over all metric names present in ``_scored_docs`` (or
        ``metric_results`` if provided) and looks up the ``Metric`` object
        for aggregation/stderr when available, falling back to ``mean``
        for unknown metrics.

        Returns ``(agg_metrics, sample_len)`` where keys are in
        ``"metric,{self.name}"`` / ``"metric_stderr,{self.name}"`` format.
        """
        from lm_eval.api.metrics import mean, stderr_for_metric

        agg: dict[str, Any] = {}
        sample_len = 0

        # Resolve values once — no branching on metric_results below.
        if metric_results is not None:
            results = metric_results
        else:
            results: dict[str, list] = {}
            for sd in self._scored_docs.values():
                for mn, val in sd.reduced_scores.items():
                    results.setdefault(mn, []).append(val)

        metrics_by_name = {m.name: m for m in self.metrics or []}

        for metric_name, values in results.items():
            if not values:
                continue
            sample_len = max(sample_len, len(values))

            m = metrics_by_name.get(metric_name)
            if m is not None:
                agg_fn = m.aggregation
                if agg_fn is not None:
                    agg[str(MetricKey(metric_name, self.name))] = m.aggregate(values)
                else:
                    eval_logger.warning(
                        f"No aggregation function for metric '{metric_name}' in scorer '{self.name}'. "
                        f"Falling back to mean. This may produce incorrect results for corpus-level metrics."
                    )
                    agg_fn = mean
                    agg[str(MetricKey(metric_name, self.name))] = mean(values)
            else:
                # Unknown metric (e.g. from process_results): default to mean
                eval_logger.warning(
                    f"No aggregation function for metric '{metric_name}' in scorer '{self.name}'. "
                    f"Falling back to mean. This may produce incorrect results for corpus-level metrics."
                )
                agg_fn = mean
                agg[str(MetricKey(metric_name, self.name))] = mean(values)

            # Stderr
            stderr_key = str(MetricKey(metric_name, self.name, is_stderr=True))
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
