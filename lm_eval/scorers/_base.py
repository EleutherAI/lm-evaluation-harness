from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from typing_extensions import Self


if TYPE_CHECKING:
    from collections.abc import Mapping

    from lm_eval.api.filter import Filter, FilterEnsemble
    from lm_eval.api.instance import Instance
    from lm_eval.api.metrics import Metric

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

    @property
    def parent_metric(self) -> str | None:
        """Extract parent from composite names: ``'pass@1(exact_match)'`` → ``'exact_match'``."""
        m = self.metric
        if m.endswith(")") and "(" in m:
            _, _, parent = m.partition("(")
            return parent[:-1]  # strip trailing ")"
        return None

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
    """Base scorer defining the filter → score → reduce → aggregate pipeline.

    Subclass hooks
    ~~~~~~~~~~~~~~
    * **score_doc** — override to define per-document scoring logic.
    * **default_filter_cfg** — list of filter configs (dicts) or ``Filter``
      subclasses to use when no explicit filter is provided.
    * **default_metric_cfg** — list of metric configs (dicts) or ``Metric``
      instances to use when no explicit metrics are provided.

    Precedence (highest → lowest):

    1. Explicit ``cfg["filter"]`` / ``cfg["metric_list"]`` passed to ``from_dict``
    2. ``cls.default_filter_cfg`` / ``cls.default_metric_cfg``
    3. Hardcoded fallback (``take_first`` / *global_metrics*)
    """

    # -- Subclass-overridable class defaults (not instance fields) ---------
    # Each entry can be a config dict (same format as YAML) or an actual
    # Filter class / Metric instance for inline definitions.
    default_filter_cfg: ClassVar[list[dict[str, Any] | type[Filter]] | None] = None
    default_metric_cfg: ClassVar[list[dict[str, Any] | Metric] | None] = None

    # -- Instance fields ---------------------------------------------------
    name: str
    filter: FilterEnsemble
    metrics: list[Metric] | None = None
    context: dict[str, Any] = field(default_factory=dict)
    _scored_docs: dict[int, ScoredDoc] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
        global_metrics: list[Metric] | None = None,
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
        global_metrics = global_metrics or []

        # --- build filter ensemble ---
        # Precedence: explicit cfg > class default > take_first fallback
        filter_name = cfg.get("name", "none")
        filter_cfg = cfg.get("filter")
        if not filter_cfg and cls.default_filter_cfg is not None:
            filter_cfg = cls.default_filter_cfg
        elif not filter_cfg:
            filter_cfg = [{"function": "take_first"}]
        filter_ensemble = cls._resolve_filters(filter_name, filter_cfg)

        # --- build metrics ---
        # Precedence: explicit cfg > class default > global_metrics fallback
        if cfg.get("metric_list"):
            metrics = cls._resolve_metrics(cfg["metric_list"])
        elif cls.default_metric_cfg is not None:
            metrics = cls._resolve_metrics(cls.default_metric_cfg)
        else:
            metrics = list(global_metrics)

        return cls(
            name=filter_name,
            filter=filter_ensemble,
            metrics=metrics,
        )

    @classmethod
    def default_scorer(cls, global_metrics: list[Metric]) -> Self:
        """Build the default scorer with the given metrics.

        Filter defaults to ``cls.default_filter_cfg`` if set, otherwise
        ``take_first``.
        """
        return cls.from_dict({"name": "none"}, global_metrics=global_metrics)

    # ------------------------------------------------------------------
    # Resolvers (override for fully custom construction)
    # ------------------------------------------------------------------

    @classmethod
    def _resolve_filters(
        cls, filter_name: str, filter_cfg: list[dict[str, Any] | type[Filter]]
    ) -> FilterEnsemble:
        """Build a :class:`FilterEnsemble` from a mixed list.

        Each entry in *filter_cfg* may be:

        * A **dict** — ``{"function": "registered_name", ...kwargs}``
          (looked up from the filter registry).
        * A **Filter subclass** — used directly as a factory.
        """
        from functools import partial

        from lm_eval.api.filter import Filter, FilterEnsemble as FEns
        from lm_eval.filters import get_filter

        filters: list = []
        for item in filter_cfg:
            if isinstance(item, dict):
                fn_name = item["function"]
                kwargs = {k: v for k, v in item.items() if k != "function"}
                filter_cls = get_filter(fn_name)
                filters.append(partial(filter_cls, **kwargs) if kwargs else filter_cls)
            elif isinstance(item, type) and issubclass(item, Filter):
                filters.append(item)
            else:
                raise TypeError(
                    f"Filter config entries must be dicts or Filter subclasses, "
                    f"got {type(item).__name__}: {item!r}"
                )
        return FEns(name=filter_name, filters=filters)

    @classmethod
    def _resolve_metrics(
        cls, metric_cfg: list[dict[str, Any] | Metric]
    ) -> list[Metric]:
        """Build a list of :class:`Metric` objects from a mixed list.

        Each entry in *metric_cfg* may be:

        * A **dict** — ``{"metric": "registered_name", ...}``
          (resolved via ``Metric.from_dict``).
        * A **Metric instance** — used as-is.
        """
        from lm_eval.api.metrics import Metric

        metrics: list[Metric] = []
        for item in metric_cfg:
            if isinstance(item, Metric):
                metrics.append(item)
            elif isinstance(item, dict):
                metrics.append(Metric.from_dict(item))
            else:
                raise TypeError(
                    f"Metric config entries must be dicts or Metric instances, "
                    f"got {type(item).__name__}: {item!r}"
                )
        return metrics

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

        Delegates per-document scoring to :meth:`score_doc`, which subclasses
        must implement.
        """
        return {
            doc_id: self.score_doc(doc_id, doc_instances)
            for doc_id, doc_instances in instances.items()
        }

    def score_doc(self, doc_id: int, doc_instances: list[Instance]) -> ScoredDoc:
        """Score a single document's instances. Subclasses must implement this.

        Override this method to define custom scoring logic.  Use
        :meth:`_dispatch_metrics` as a helper to run configured metrics,
        or compute scores directly and return a :class:`ScoredDoc`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement score_doc(). "
            "Use GenScorer for generate_until tasks or LLScorer for loglikelihood tasks."
        )

    def _dispatch_metrics(
        self,
        references: Any,
        predictions: Any,
        metric_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call each Metric.compute(references, predictions) and collect per-doc results.

        Extra kwargs are forwarded to each metric function from two sources
        (later sources override earlier ones):

        1. ``self.context`` — task-level runtime config (e.g. ``multiple_targets``).
        2. *metric_kwargs* — per-instance overrides from ``Instance.metadata["metric_kwargs"]``.

        ``filter_kwargs`` inside ``Metric.compute`` ensures only parameters
        the function actually accepts are passed through.
        """
        result_dict: dict[str, Any] = {}
        if not self.metrics:
            return result_dict
        ctx = {**self.context, **(metric_kwargs or {})}
        for m in self.metrics:
            score = m.compute(references, predictions, **ctx)
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
                if len(values) == 1:
                    # No reduction needed for single-value metrics
                    sd.reduced_scores[metric_name] = values[0]
                    continue
                m = metrics_by_name.get(metric_name)
                if m is not None and m.reduction is not None:
                    res = m.reduction(sd.reference, values)
                    if isinstance(res, dict):
                        # If reduction returns multiple sub-metrics, flatten them into reduced_scores
                        for sub_metric_name, sub_value in res.items():
                            full_sub_metric_name = f"{sub_metric_name}({metric_name})"
                            sd.reduced_scores[full_sub_metric_name] = sub_value
                    else:
                        sd.reduced_scores[metric_name] = res
                else:
                    # Unknown metric (e.g. from process_results): take first
                    if len(values) > 1:
                        eval_logger.warning(
                            "No reduction function for metric '%s' in scorer '%s'. Falling back to first value.",
                            metric_name,
                            self.name,
                        )
                    sd.reduced_scores[metric_name] = values[0]

    def aggregate(
        self,
        metric_results: dict[str, list] | None = None,
        bootstrap_iters: int | None = 100000,
        aggregation_overrides: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], int]:
        """Aggregate metric results and compute stderr.

        Iterates over all metric names present in ``_scored_docs`` (or
        ``metric_results`` if provided) and looks up the ``Metric`` object
        for aggregation/stderr when available.  When *aggregation_overrides*
        is supplied (legacy Python tasks that override ``Task.aggregation()``),
        those functions take precedence over the ``mean`` fallback for metrics
        not covered by a ``Metric`` object.

        Returns ``(agg_metrics, sample_len)`` where keys are in
        ``"metric,{self.name}"`` / ``"metric_stderr,{self.name}"`` format.
        """
        from lm_eval.api.metrics import mean, stderr_for_metric

        agg: dict[str, Any] = {}
        sample_len = 0

        # Resolve values once
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
            # Fall back to parent for composite keys like "pass@1(exact_match)"
            if m is None:
                parent = MetricKey(metric_name, self.name).parent_metric
                if parent is not None:
                    m = metrics_by_name.get(parent)
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
            elif aggregation_overrides and metric_name in aggregation_overrides:
                agg_fn = aggregation_overrides[metric_name]
                agg[str(MetricKey(metric_name, self.name))] = agg_fn(values)
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
    # Data access / serialisation
    # ------------------------------------------------------------------

    @property
    def scored_docs(self) -> Mapping[int, ScoredDoc]:
        return self._scored_docs

    def export_reduced(self) -> dict[str, list]:
        """Export {metric_name: [per_doc_values]} from reduced_scores."""
        metrics: dict[str, list] = {}
        for sd in self._scored_docs.values():
            for mn, val in sd.reduced_scores.items():
                metrics.setdefault(mn, []).append(val)
        return metrics

    def import_reduced(self, metric_data: dict[str, list]) -> None:
        """Rebuild _scored_docs from flat metric lists (after distributed gather)."""
        n_docs = max((len(v) for v in metric_data.values()), default=0)
        self._scored_docs = {}
        for i in range(n_docs):
            reduced = {mn: vals[i] for mn, vals in metric_data.items() if i < len(vals)}
            self._scored_docs[i] = ScoredDoc(
                doc_id=i, reference=None, scores={}, reduced_scores=reduced
            )

    def set_results(self, scored_docs: dict[int, ScoredDoc]) -> None:
        """Store scored documents and apply reduction."""
        self._scored_docs = scored_docs
        self.reduce(scored_docs)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def higher_is_better(self) -> dict[str, bool]:
        """Return ``{metric_name: bool}`` for all metrics in this scorer."""
        base = {m.name: m.higher_is_better for m in (self.metrics or [])}
        # Inherit higher_is_better for composite keys from their parent metric
        if self._scored_docs:
            for sd in self._scored_docs.values():
                for key in sd.reduced_scores:
                    if key not in base:
                        mk = MetricKey(key, self.name)
                        if mk.parent_metric and mk.parent_metric in base:
                            base[key] = base[mk.parent_metric]
                break  # all docs share the same metric names
        return base


# ---------------------------------------------------------------------------
# Concrete scorer subclasses
# ---------------------------------------------------------------------------


@dataclass
class GenScorer(Scorer):
    """Scorer for ``generate_until`` tasks.

    Each repeat is scored independently, producing ``list[T]`` per doc per
    metric.  Subclass this to create custom generation evaluation pipelines
    — override :meth:`score_doc` and optionally use :meth:`_dispatch_metrics`.
    """

    def score_doc(self, doc_id: int, doc_instances: list[Instance]) -> ScoredDoc:
        inst = doc_instances[0]  # 1 instance per doc for generate_until
        resps: list[str] = inst.filtered_resps[self.name]  # [str * R]
        target = inst.target
        metric_kwargs = inst.metadata.get("metric_kwargs")

        repeat_scores: dict[str, list[Any]] = self._dispatch_metrics(
            [target], resps, metric_kwargs=metric_kwargs
        )
        return ScoredDoc(
            doc_id=doc_id,
            reference=target,
            scores=dict(repeat_scores),
        )


@dataclass
class LLScorer(Scorer):
    """Scorer for ``loglikelihood`` / ``loglikelihood_rolling`` / ``multiple_choice`` tasks.

    Repeats are always 1.  The scalar metric result is wrapped in a
    single-element list so that the downstream ``reduce`` step works
    uniformly with :class:`GenScorer`.
    """

    def score_doc(self, doc_id: int, doc_instances: list[Instance]) -> ScoredDoc:
        from lm_eval.api._metrics.results import LLResults

        metric_kwargs = doc_instances[0].metadata.get("metric_kwargs")
        results_obj = LLResults.from_instances(doc_instances)
        references = results_obj.targets
        per_doc = self._dispatch_metrics(
            references, results_obj, metric_kwargs=metric_kwargs
        )
        return ScoredDoc(
            doc_id=doc_id,
            reference=references,
            scores={mn: [v] for mn, v in per_doc.items()},
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_scorer(
    cfg: dict[str, Any] | None = None,
    global_metrics: list[Metric] | None = None,
    output_type: str | None = None,
) -> Scorer:
    """Construct the appropriate scorer subclass based on *output_type*.

    Uses :class:`GenScorer` for ``generate_until`` tasks, and
    :class:`LLScorer` for all other output types.
    """
    cls = GenScorer if output_type == "generate_until" else LLScorer
    if cfg is not None:
        return cls.from_dict(cfg, global_metrics=global_metrics)
    return cls.default_scorer(global_metrics or [])
