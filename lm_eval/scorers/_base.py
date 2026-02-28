from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar
from typing_extensions import Self

from ._types import MetricKey, ScoredDoc


if TYPE_CHECKING:
    from collections.abc import Mapping

    from lm_eval.api.filter import Filter, FilterEnsemble
    from lm_eval.api.instance import Instance
    from lm_eval.api.metrics import Metric

eval_logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class Scorer:
    """Base scorer defining the filter → score → reduce → aggregate pipeline.

    For generation tasks, subclass :class:`GenScorer` which offers two
    tiers of extensibility (from simplest to most control):

    1. **Config** — set ``default_filter_cfg`` / ``default_metric_cfg``
       class variables.  No scoring code needed.
    2. **Per-doc** — override ``GenScorer.score(reference, predictions)``
       to return ``{metric: [scores]}``.  No ``Instance`` knowledge needed.

    For full control (e.g. batch scoring), override :meth:`score_instances`.

    Filter / metric precedence (highest → lowest):

    1. Explicit ``cfg["filter"]`` / ``cfg["metric_list"]`` passed to ``from_dict``
    2. ``cls.default_filter_cfg`` / ``cls.default_metric_cfg``
    3. Hardcoded fallback (``noop`` / *global_metrics*)
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
    _scored_docs: dict[int, ScoredDoc] = field(
        default_factory=dict, init=False, repr=False
    )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
        *,
        output_type: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """Build a Scorer from a config dict.

        Expected shape::

            {
                "name": "strict-match",
                "filter": [
                    {"function": "regex", "regex_pattern": "..."},
                    {"function": "take_first"},
                ],
                "metric_list": [
                    {"metric": "exact_match", "aggregation": "mean", ...},
                ],
            }

        *output_type* is used as a last-resort fallback when neither the
        config nor the class provides metrics (see
        ``DEFAULT_METRIC_REGISTRY``).

        Any extra *kwargs* are forwarded to the constructor (e.g., custom
        dataclass fields on scorer subclasses).
        """
        name = cfg.get("name", "none")
        return cls(
            name=name,
            filter=cls._build_filter(name, cfg),
            metrics=cls._build_metrics(cfg, output_type=output_type),
            **kwargs,
        )

    @classmethod
    def _build_filter(cls, name: str, cfg: dict[str, Any]) -> FilterEnsemble:
        """Resolve filter config: explicit cfg > ClassVar default > noop fallback."""
        filter_cfg = (
            cfg.get("filter") or cls.default_filter_cfg or [{"function": "noop"}]
        )
        return cls._resolve_filters(name, filter_cfg)

    @classmethod
    def _build_metrics(
        cls,
        cfg: dict[str, Any],
        *,
        output_type: str | None = None,
    ) -> list[Metric]:
        """Resolve metric config with a clear 3-tier precedence.

        1. Explicit ``cfg["metric_list"]`` (per-pipeline or task-level)
        2. ``cls.default_metric_cfg`` (scorer class defaults)
        3. ``DEFAULT_METRIC_REGISTRY`` based on *output_type*
        """
        if cfg.get("metric_list"):
            return cls._resolve_metrics(cfg["metric_list"], output_type or "")
        if cls.default_metric_cfg is not None:
            return cls._resolve_metrics(cls.default_metric_cfg, output_type or "")
        if output_type is not None:
            from lm_eval.api.registry import DEFAULT_METRIC_REGISTRY

            defaults = DEFAULT_METRIC_REGISTRY.get(output_type, [])
            if defaults:
                return cls._resolve_metrics(
                    [{"metric": name} for name in defaults], output_type
                )
        return []

    @classmethod
    def default_scorer(cls, name: str = "none", **kwargs: Any) -> Self:
        """Build the default scorer (no explicit config).

        Filter defaults to ``cls.default_filter_cfg`` if set, otherwise
        ``noop``.
        """
        return cls.from_dict({"name": name}, **kwargs)

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
                kwargs = item.get("kwargs", {})
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
        cls, metric_cfg: list[dict[str, Any] | Metric], output_type: str
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
                metrics.append(Metric.from_dict(item, output_type))
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
        self, instances: Mapping[int, list[Instance]]
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
            "Override score_doc() on your Scorer subclass, "
            "or subclass GenScorer and override score() for per-doc scoring, "
            "or override score_instances() for batch scoring."
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
                    eval_logger.warning(
                        "No reduction function for metric '%s' in scorer '%s'. "
                        "Falling back to first value.",
                        metric_name,
                        self.name,
                    )
                    sd.reduced_scores[metric_name] = values[0]

    def aggregate(
        self,
        metric_results: Mapping[str, list[Any]] | None = None,
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
        results = (
            metric_results if metric_results is not None else self.export_reduced()
        )
        metrics_by_name = {m.name: m for m in self.metrics or []}

        for metric_name, values in results.items():
            if not values:
                continue
            sample_len = max(sample_len, len(values))

            # Resolve metric object (check parent for composite keys like "pass@1(exact_match)")
            m = metrics_by_name.get(metric_name)
            if m is None and (
                parent := MetricKey(metric_name, self.name).parent_metric
            ):
                m = metrics_by_name.get(parent)

            # Resolve aggregation function: metric > legacy override > mean fallback
            if m is not None and m.aggregation is not None:
                agg_fn = m.aggregation
            elif aggregation_overrides and metric_name in aggregation_overrides:
                agg_fn = aggregation_overrides[metric_name]
            else:
                eval_logger.warning(
                    "No aggregation function for metric '%s' in scorer '%s'. "
                    "Falling back to mean. This may produce incorrect results "
                    "for corpus-level metrics.",
                    metric_name,
                    self.name,
                )
                agg_fn = mean

            # Aggregate + stderr
            key = str(MetricKey(metric_name, self.name))
            agg[key] = agg_fn(values)

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

    Extensibility hooks (from simplest to most control)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    **Tier 1 — Config-only**: Set ``default_filter_cfg`` and/or
    ``default_metric_cfg`` class variables.  No scoring code needed.

    **Tier 2 — Per-doc scoring**: Override :meth:`score` to define custom
    scoring as ``(reference, predictions) → {metric: [scores]}``.

    **Full control**: Override :meth:`score_instances` for batch scoring
    (e.g. batched LLM judge calls, code sandbox pools).  Use
    :meth:`_extract_inputs` to pull ``(reference, predictions, metric_kwargs)``
    from each document's instances.

    The default call chain is::

        score_instances()  →  score_doc()  →  score()

    Example batch scorer overriding ``score_instances``::

        @register_scorer("ai_judge")
        @dataclass
        class AIJudgeScorer(GenScorer):
            judge_model: str = "claude-sonnet-4-6"

            def score_instances(self, instances):
                inputs = {did: self._extract_inputs(insts)
                          for did, insts in instances.items()}
                ratings = batch_judge(self.judge_model,
                    {did: (ref, preds[0])
                     for did, (ref, preds, _) in inputs.items()})
                return {
                    did: ScoredDoc(
                        doc_id=did, reference=ref,
                        scores={"judge": [ratings[did]]})
                    for did, (ref, preds, _) in inputs.items()
                }
    """

    # ------------------------------------------------------------------
    # Scoring hooks
    # ------------------------------------------------------------------

    def score_doc(self, doc_id: int, doc_instances: list[Instance]) -> ScoredDoc:
        """Extract inputs from a document's instances and delegate to :meth:`score`."""
        ref, preds, mkw = self._extract_inputs(doc_instances)
        return ScoredDoc(
            doc_id=doc_id,
            reference=ref,
            scores=self.score(ref, preds, metric_kwargs=mkw),
        )

    def score(
        self,
        reference: str | list[str],
        predictions: list[str],
        metric_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, list[float]]:
        """Per-document scoring.  Override for custom generation scoring.

        This is the simplest hook.  Receives clean inputs and returns
        metric scores — no need to work with ``Instance`` or ``ScoredDoc``.

        Args:
            reference: The gold answer(s).
            predictions: Model predictions (one per repeat).
            metric_kwargs: Optional per-instance metric overrides.

        Returns:
            ``{metric_name: [score_per_repeat]}``.

        Example::

            @register_scorer("code_exec")
            @dataclass
            class CodeExecScorer(GenScorer):
                timeout: int = 10

                def score(self, reference, predictions, **kwargs):
                    return {"pass": [
                        1.0 if run(code, self.timeout) == reference
                        else 0.0
                        for code in predictions
                    ]}
        """
        return self._dispatch_metrics(
            [reference], predictions, metric_kwargs=metric_kwargs
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_inputs(
        self, doc_instances: list[Instance]
    ) -> tuple[Any, list[str], dict[str, Any] | None]:
        """Extract ``(reference, predictions, metric_kwargs)`` from a doc's instances."""
        inst = doc_instances[0]
        if self.name not in inst.filtered_resps:
            raise KeyError(
                f"Scorer '{self.name}' not found in filtered_resps. "
                f"Available: {list(inst.filtered_resps.keys())}. "
                f"Was apply_filters() called?"
            )
        return (
            inst.target,
            inst.filtered_resps[self.name],
            inst.metadata.get("metric_kwargs", {}),
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
        results_obj = LLResults.from_instances(doc_instances, self.name)
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
    output_type: str | None = None,
    scorer_type: str | dict[str, Any] | None = None,
) -> Scorer:
    """Construct the appropriate scorer subclass.

    *scorer_type* can be:

    * **str** — scorer name, resolved from the scorer registry
      (e.g. ``"first_token"`` → :class:`FirstTokenScorer`).
    * **dict** — ``{"type": "scorer_name", ...kwargs}`` where extra
      keys are forwarded to the scorer constructor as kwargs.
    * **None** — fall back to :class:`GenScorer` / :class:`LLScorer`
      based on *output_type*.

    Metrics are resolved inside ``Scorer.from_dict`` with a 3-tier
    precedence: cfg > scorer class default > DEFAULT_METRIC_REGISTRY.
    """
    scorer_kwargs: dict[str, Any] = {}
    scorer_name: str | None = None

    if isinstance(scorer_type, dict):
        scorer_name = scorer_type["type"]
        scorer_kwargs = {k: v for k, v in scorer_type.items() if k != "type"}
    elif isinstance(scorer_type, str):
        scorer_name = scorer_type

    if scorer_name is not None:
        from lm_eval.api.registry import get_scorer

        cls = get_scorer(scorer_name)
    elif output_type == "generate_until":
        cls = GenScorer
    elif output_type in (
        "loglikelihood",
        "loglikelihood_rolling",
        "multiple_choice",
    ):
        cls = LLScorer
    else:
        raise ValueError(
            f"Cannot infer scorer for output_type={output_type!r}. "
            f"Pass an explicit scorer_type or use a known output_type."
        )

    if cfg is not None:
        return cls.from_dict(cfg, output_type=output_type, **scorer_kwargs)
    if scorer_kwargs:
        return cls.from_dict(
            {"name": scorer_name}, output_type=output_type, **scorer_kwargs
        )
    if scorer_name is not None:
        return cls.default_scorer(name=scorer_name, output_type=output_type)
    return cls.default_scorer(output_type=output_type)
