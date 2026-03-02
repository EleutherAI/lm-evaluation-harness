from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar
from typing_extensions import Self, TypedDict

from ._types import MetricKey, ScoredDoc


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lm_eval.api.filter import Filter, FilterEnsemble
    from lm_eval.api.instance import Instance
    from lm_eval.api.metrics import Metric
    from lm_eval.config.task import FilterStep, MetricConfig, ScorerConfig

    from ._types import ReducedDoc

eval_logger = logging.getLogger(__name__)


class _ScorerCfg(TypedDict):
    """Normalised per-pipeline config consumed by [Scorer.from_dict][Scorer.from_dict].

    Each ``TaskConfig.filter_list`` entry becomes one ``_ScorerCfg`` passed
    to [build_scorer][build_scorer].  After ``TaskConfig._normalize_scoring_config()``,
    all three keys are guaranteed present:

    * ``name`` — pipeline identifier.
    * ``filter`` — filter steps, or ``[]`` to fall back to the scorer's
      ``default_filter_cfg`` (→ ``noop``).
    * ``metric_list`` — metric configs, or ``[]`` to fall back to
      ``default_metric_cfg`` (→ ``DEFAULT_METRIC_REGISTRY``).
    """

    name: str
    """Pipeline identifier (e.g. ``"strict-match"``, ``"none"``)."""

    filter: list[FilterStep]
    """Ordered filter-step configs.  Each entry follows the
    [FilterStep][lm_eval.config.task.FilterStep] shape (``"function"`` key
    plus optional ``"kwargs"``).
    An empty list ``[]`` signals "no explicit filters" — ``Scorer._build_filter``
    falls back to the scorer's ``default_filter_cfg`` → ``[{"function": "noop"}]``."""

    metric_list: list[MetricConfig]
    """Per-pipeline metric configs.  Each entry follows the
    [MetricConfig][lm_eval.config.task.MetricConfig] shape (``"metric"`` key
    plus optional aggregation/kwargs fields).
    An empty list ``[]`` signals "no explicit metrics" — ``Scorer._build_metrics``
    falls back to ``default_metric_cfg`` → ``DEFAULT_METRIC_REGISTRY``."""


@dataclass(kw_only=True)
class Scorer:
    """Base scorer defining the filter → score → reduce → aggregate pipeline.

    For generation tasks, subclass [GenScorer][GenScorer] which offers two
    tiers of extensibility (from simplest to most control):

    1. **Config** — set ``default_filter_cfg`` / ``default_metric_cfg``
       class variables.  No scoring code needed.
    2. **Per-doc** — override ``GenScorer.score(reference, predictions)``
       to return ``{metric: [scores]}``.  No ``Instance`` knowledge needed.

    For full control (e.g. batch scoring), override [score_instances][score_instances].

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
    _raw_docs: dict[int, ScoredDoc] = field(
        default_factory=dict, init=False, repr=False
    )
    _reduced_docs: dict[int, ReducedDoc] = field(
        default_factory=dict, init=False, repr=False
    )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(
        cls,
        cfg: _ScorerCfg,
        *,
        output_type: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """Build a Scorer from a normalised pipeline config.

        *cfg* is a [_ScorerCfg][_ScorerCfg] produced by
        ``TaskConfig._normalize_scoring_config()``:

            ```python
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
            ```

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
    def _build_filter(cls, name: str, cfg: _ScorerCfg) -> FilterEnsemble:
        """Resolve filter config: explicit cfg > ClassVar default > noop fallback."""
        if cfg.get("filter"):
            return cls._resolve_filters(name, cfg["filter"])
        if cls.default_filter_cfg:
            return cls._resolve_filters(name, cls.default_filter_cfg)
        return cls._resolve_filters(name, [{"function": "noop"}])

    @classmethod
    def _build_metrics(
        cls,
        cfg: _ScorerCfg,
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
        return cls.from_dict({"name": name, "filter": [], "metric_list": []}, **kwargs)

    # ------------------------------------------------------------------
    # Resolvers (override for fully custom construction)
    # ------------------------------------------------------------------

    @classmethod
    def _resolve_filters(
        cls,
        filter_name: str,
        filter_cfg: Sequence[dict[str, Any] | type[Filter]] | Sequence[FilterStep],
    ) -> FilterEnsemble:
        """Build a [FilterEnsemble][lm_eval.api.filter.FilterEnsemble] from a mixed list.

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
        cls,
        metric_cfg: Sequence[dict[str, Any] | Metric] | Sequence[MetricConfig],
        output_type: str,
    ) -> list[Metric]:
        """Build a list of [Metric][lm_eval.api.metrics.Metric] objects from a mixed list.

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

        Delegates per-document scoring to [score_doc][.score_doc], which subclasses
        must implement.
        """
        return {
            doc_id: self.score_doc(doc_id, doc_instances)
            for doc_id, doc_instances in instances.items()
        }

    def score_doc(self, doc_id: int, doc_instances: list[Instance]) -> ScoredDoc:
        """Score a single document's instances. Subclasses must implement this.

        Override this method to define custom scoring logic.  Use
        [_dispatch_metrics][._dispatch_metrics] as a helper to run configured metrics,
        or compute scores directly and return a [ScoredDoc][lm_eval.scorers.ScoredDoc].
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

    def reduce(self, scored_docs: dict[int, ScoredDoc]) -> dict[int, ReducedDoc]:
        """Reduce per-doc ``list[T]`` → ``T`` for each document.

        Pure function: takes [ScoredDoc][lm_eval.scorers.ScoredDoc] objects (immutable raw scores)
        and returns ``{doc_id: {metric: scalar}}`` dicts ready for aggregation.

        For each metric in each document:

        * **Single value** — passed through as-is (no reduction needed).
        * **Multiple values + reduction fn** — calls ``Metric.reduction(reference, values)``.
          If the reduction returns a dict, composite keys like ``"pass@1(metric)"``
          are created.
        * **No reduction fn** — warns and takes the first value.
        """
        metrics_by_name = self._metrics_by_name
        result: dict[int, ReducedDoc] = {}

        for sd in scored_docs.values():
            values_dict: ReducedDoc = {}
            for metric_name, score_list in sd.scores.items():
                if len(score_list) == 1:
                    values_dict[metric_name] = score_list[0]
                    continue
                m = metrics_by_name.get(metric_name)
                if m is not None and m.reduction is not None:
                    res = m.reduction(sd.reference, score_list)
                    if isinstance(res, dict):
                        for sub_metric_name, sub_value in res.items():
                            values_dict[f"{sub_metric_name}({metric_name})"] = sub_value
                    else:
                        values_dict[metric_name] = res
                else:
                    raise ValueError(
                        f"Metric '{metric_name}' in scorer '{self.name}' has "
                        f"{len(score_list)} values per document (repeats > 1) "
                        f"but no reduction function is configured. Set a "
                        f"reduction (e.g., 'take_first', 'pass@k', 'mean') in "
                        f"your metric config, or set repeats to 1."
                    )

            result[sd.doc_id] = values_dict

        return result

    def aggregate(
        self,
        reduced_docs: Mapping[int, ReducedDoc],
        bootstrap_iters: int | None = 100000,
        aggregation_overrides: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], int]:
        """Aggregate reduced docs and compute stderr.

        Pure function: takes ``{doc_id: {metric: value}}`` and produces
        aggregated ``"metric,scorer"`` keyed results.  When
        *aggregation_overrides* is supplied (legacy Python tasks that override
        ``Task.aggregation()``), those functions take precedence over the
        ``mean`` fallback for metrics not covered by a ``Metric`` object.

        Returns ``(agg_metrics, sample_len)`` where keys are in
        ``"metric,{self.name}"`` / ``"metric_stderr,{self.name}"`` format.
        """
        from lm_eval.api.metrics import mean, stderr_for_metric

        # Transpose doc-first → metric-first: {metric: [values]}
        results: dict[str, list[float]] = {}
        for rd in reduced_docs.values():
            for mn, val in rd.items():
                results.setdefault(mn, []).append(val)

        agg: dict[str, Any] = {}
        sample_len = 0
        metrics_by_name = self._metrics_by_name

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
                eval_logger.error(
                    "No aggregation function for metric '%s' in scorer '%s'. "
                    "Defaulting to 'mean'. WARNING: this will produce INCORRECT "
                    "results for corpus-level metrics (BLEU, perplexity, F1, etc.). "
                    "Set 'aggregation' explicitly in your metric config.",
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
    def raw_docs(self) -> Mapping[int, ScoredDoc]:
        """Per-document raw scoring results (pre-reduction).

        Empty after [import_reduced][.import_reduced] — raw scores only exist on the
        rank that performed scoring.
        """
        return self._raw_docs

    @property
    def reduced_docs(self) -> Mapping[int, ReducedDoc]:
        """Per-document reduced results (post-reduction), ready for aggregation."""
        return self._reduced_docs

    def export_reduced(self) -> dict[int, ReducedDoc]:
        """Export ``{doc_id: {metric: value}}`` for distributed gathering.

        Since ``ReducedDoc`` is a plain ``dict[str, float]``, this is a
        shallow copy.  Merge across ranks is a simple ``dict.update``
        since doc IDs are unique per rank.
        """
        return dict(self._reduced_docs)

    def import_reduced(self, doc_data: dict[int, ReducedDoc]) -> None:
        """Import merged results after distributed gather.

        Raw scores are not available after import (they live on the
        source ranks).
        """
        self._raw_docs = {}
        self._reduced_docs = dict(doc_data)

    def set_results(self, scored_docs: dict[int, ScoredDoc]) -> None:
        """Store raw scored documents and compute reduction."""
        self._raw_docs = scored_docs
        self._reduced_docs = self.reduce(scored_docs)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def _metrics_by_name(self) -> dict[str, Metric]:
        """Lookup table: metric name → Metric object."""
        return {m.name: m for m in self.metrics or []}

    @property
    def higher_is_better(self) -> dict[str, bool]:
        """Return ``{metric_name: bool}`` for all metrics in this scorer."""
        base = {m.name: m.higher_is_better for m in (self.metrics or [])}
        # Inherit higher_is_better for composite keys from their parent metric
        if self._reduced_docs:
            for rd in self._reduced_docs.values():
                for key in rd:
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

    **Tier 2 — Per-doc scoring**: Override [score][.score] to define custom
    scoring as ``(reference, predictions) → {metric: [scores]}``.

    **Full control**: Override [score_instances][.score_instances] for batch scoring
    (e.g. batched LLM judge calls, code sandbox pools).  Use
    [_extract_inputs][._extract_inputs] to pull ``(reference, predictions, metric_kwargs)``
    from each document's instances.

    The default call chain is:
    ``score_instances()`` → ``score_doc()`` → ``score()``

    Example:
        ```python
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
        ```
    """

    # ------------------------------------------------------------------
    # Scoring hooks
    # ------------------------------------------------------------------

    def score_doc(self, doc_id: int, doc_instances: list[Instance]) -> ScoredDoc:
        """Extract inputs from a document's instances and delegate to [score][.score]."""
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

        Example:
            ```python
            @register_scorer("code_exec")
            @dataclass
            class CodeExecScorer(GenScorer):
                timeout: int = 10

                def score(self, reference, predictions, **kwargs):
                    return {
                        "pass": [
                            1.0 if run(code, self.timeout) == reference else 0.0
                            for code in predictions
                        ]
                    }
            ```
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
    uniformly with [GenScorer][GenScorer].
    """

    def score_doc(self, doc_id: int, doc_instances: list[Instance]) -> ScoredDoc:
        from lm_eval.api.metrics.results import LLResults

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
    cfg: _ScorerCfg | None = None,
    output_type: str | None = None,
    scorer_type: str | ScorerConfig | None = None,
) -> Scorer:
    """Construct the appropriate scorer subclass.

    *cfg* is a [_ScorerCfg][_ScorerCfg] (normalised pipeline config from
    ``TaskConfig.filter_list``).

    *scorer_type* can be:

    * **str** — scorer name, resolved from the scorer registry
      (e.g. ``"first_token"`` → [FirstTokenScorer][lm_eval.scorers.extraction.FirstTokenScorer]).
    * [ScorerConfig][lm_eval.config.task.ScorerConfig] **dict** —
      ``{"type": "scorer_name", ...kwargs}`` where extra keys are
      forwarded to the scorer constructor as kwargs.
    * **None** — fall back to [GenScorer][GenScorer] / [LLScorer][LLScorer]
      based on *output_type*.

    Metrics are resolved inside ``Scorer.from_dict`` with a 3-tier
    precedence: cfg > scorer class default > DEFAULT_METRIC_REGISTRY.
    """
    scorer_kwargs: dict[str, Any] = {}
    scorer_name: str | None = None

    if isinstance(scorer_type, dict):
        scorer_name = scorer_type["type"]
        scorer_kwargs = scorer_type.get("kwargs", {})
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

    if cfg is None:
        cfg = {"name": scorer_name or "none", "filter": [], "metric_list": []}
    return cls.from_dict(cfg, output_type=output_type, **scorer_kwargs)
