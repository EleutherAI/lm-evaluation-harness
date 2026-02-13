from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from lm_eval._types import LLResults


if TYPE_CHECKING:
    from collections.abc import Callable

    from lm_eval.api.filter import FilterEnsemble
    from lm_eval.api.instance import Instance
    from lm_eval.config.metric import Metric

eval_logger = logging.getLogger(__name__)

# Metric names that map to exact_match_mc when used in MC context
_MC_EXACT_MATCH_ALIASES = frozenset({"exact_match"})


def _group_by_doc_id(
    instances: list[Instance],
) -> dict[int, list[Instance]]:
    """Group instances by doc_id and sort each group by idx."""
    grouped: defaultdict[int, list[Instance]] = defaultdict(list)
    for inst in instances:
        grouped[inst.doc_id].append(inst)
    for group in grouped.values():
        group.sort(key=lambda x: x.idx)
    return dict(grouped)


@dataclass
class Scorer:
    name: str
    filter: FilterEnsemble
    metrics: list[Metric]
    output_type: str | None = None
    _resps: list = field(default_factory=list)

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
        global_metrics: list[Metric] | None = None,
        output_type: str | None = None,
    ) -> Scorer:
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
            metrics = [Metric.from_yaml(m) for m in cfg["metric_list"]]
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
    ) -> Scorer:
        """Build the default scorer: ``take_first`` filter with the given metrics."""
        return cls.from_dict(
            {
                "name": "none",
                "filter": [{"function": "take_first"}],
            },
            global_metrics=global_metrics,
            output_type=output_type,
        )

    # --- helper properties ---

    @property
    def metric_names(self) -> list[str]:
        return [m.name for m in self.metrics]

    @property
    def aggregation_dict(self) -> dict[str, Callable]:
        """Map metric name -> aggregation function."""
        return {
            m.name: m.aggregation for m in self.metrics if m.aggregation is not None
        }

    @property
    def higher_is_better_dict(self) -> dict[str, bool]:
        """Map metric name -> higher_is_better flag."""
        return {m.name: m.higher_is_better for m in self.metrics}

    # --- scoring pipeline ---

    def apply_filter(self, instances: list[Instance]) -> None:
        """Apply the filter to the task's instances."""
        self.filter.apply(instances)

    def score(self, instances: list[Instance]) -> dict[tuple, Any]:
        """Full scoring pipeline: filter -> group by doc_id -> compute metrics.

        Returns:
            dict mapping (metric_name, filter_key, doc_id) -> value
        """
        self.apply_filter(instances)
        grouped = _group_by_doc_id(instances)
        results: dict[tuple, Any] = {}
        for doc_id, group in grouped.items():
            filtered = [inst.filtered_resps[self.name] for inst in group]
            metric_values = self._compute_metrics(group, filtered)
            for metric_name, value in metric_values.items():
                results[(metric_name, self.name, doc_id)] = value
        return results

    def _compute_metrics(
        self, instances: list[Instance], filtered_resps: list
    ) -> dict[str, Any]:
        """Dispatch to the appropriate scoring method based on output_type."""
        if self.output_type == "loglikelihood":
            return self._score_loglikelihood(instances, filtered_resps)
        elif self.output_type == "loglikelihood_rolling":
            return self._score_loglikelihood_rolling(instances, filtered_resps)
        elif self.output_type == "multiple_choice":
            return self._score_multiple_choice(instances, filtered_resps)
        elif self.output_type == "generate_until":
            return self._score_generate_until(instances, filtered_resps)
        else:
            raise ValueError(
                f"Scorer got invalid output_type '{self.output_type}'. "
                f"Must be one of 'loglikelihood', 'loglikelihood_rolling', "
                f"'generate_until', or 'multiple_choice'."
            )

    def _dispatch_metrics(self, targets: Any, ll_results: LLResults) -> dict[str, Any]:
        """Call each Metric.compute(targets, ll_results) and collect results."""
        result_dict: dict[str, Any] = {}
        for m in self.metrics:
            score = m.compute(targets, ll_results)
            if isinstance(score, dict):
                result_dict.update(score)
            else:
                result_dict[m.name] = score
        return result_dict

    def _score_loglikelihood(
        self, instances: list[Instance], filtered_resps: list
    ) -> dict[str, Any]:
        ll, is_greedy = filtered_resps[0]
        cont = instances[0].arguments[1] if len(instances[0].arguments) > 1 else ""

        ll_results = LLResults(
            doc=instances[0].doc,
            ctx=instances[0].arguments[0],
            targets=0,
            lls=[ll],
            is_greedy=[is_greedy],
            choices=[cont],
        )
        return self._dispatch_metrics(0, ll_results)

    def _score_loglikelihood_rolling(
        self, instances: list[Instance], filtered_resps: list
    ) -> dict[str, Any]:
        (loglikelihood,) = filtered_resps
        target_text = str(instances[0].target)

        ll_results = LLResults(
            doc=instances[0].doc,
            ctx="",
            targets=0,
            lls=[loglikelihood],
            is_greedy=[False],
            choices=[target_text],
        )
        return self._dispatch_metrics(0, ll_results)

    def _score_multiple_choice(
        self, instances: list[Instance], filtered_resps: list
    ) -> dict[str, Any]:
        lls, is_greedy = zip(*filtered_resps, strict=True)

        ctx = instances[0].scoring_context
        choices = ctx["choices"]
        gold = instances[0].target

        # Handle acc_mutual_info split
        use_metric_names = {m.name for m in self.metrics}
        lls_mutual_info: list[float] = []
        if 2 * len(choices) == len(lls) and "acc_mutual_info" in use_metric_names:
            lls_unconditional = lls[len(choices) :]
            assert len(lls_unconditional) == len(choices)
            lls = lls[: len(choices)]
            lls_mutual_info = [
                ll_c - ll_u for ll_c, ll_u in zip(lls, lls_unconditional, strict=True)
            ]

        ll_results = LLResults(
            doc=instances[0].doc,
            ctx=instances[0].arguments[0],
            targets=gold,
            lls=lls,
            is_greedy=is_greedy,
            choices=choices,
            lls_mutual_info=lls_mutual_info,
        )

        # Dispatch through Metric.compute(), mapping exact_match -> exact_match_mc
        result_dict: dict[str, Any] = {}
        for m in self.metrics:
            # Map exact_match to exact_match_mc for MC context
            if m.name in _MC_EXACT_MATCH_ALIASES:
                from lm_eval.api.registry import _get_metric

                mc_metric = _get_metric("exact_match_mc")
                if mc_metric is not None:
                    score = mc_metric.compute(gold, ll_results)
                    result_dict[m.name] = score
                    continue

            score = m.compute(gold, ll_results)
            if isinstance(score, dict):
                result_dict.update(score)
            else:
                result_dict[m.name] = score
        return result_dict

    def _score_generate_until(
        self, instances: list[Instance], filtered_resps: list
    ) -> dict[str, Any]:
        gold = instances[0].target
        result = filtered_resps[0]
        ctx = instances[0].scoring_context
        multiple_target = ctx.get("multiple_target", False)

        if multiple_target:
            if not isinstance(gold, list):
                gold = [gold]

        # type casting for non-bypass metrics
        if not multiple_target:
            if type(gold) is not type(result) and not (
                any(m.name == "bypass" for m in self.metrics)
                or isinstance(result, list)
            ):
                gold = type(result)(gold)

        result_dict: dict[str, Any] = {}
        for m in self.metrics:
            metric = m.name
            if multiple_target:
                scores = []
                _gold = gold if isinstance(gold, list) else [gold]
                if metric == "exact_match":
                    result_list = [result for _ in range(len(_gold))]
                    scores = m.compute(
                        references=_gold,
                        predictions=result_list,
                    )[metric]
                    result_score = 1.0 if scores > 0.0 else 0.0
                else:
                    for gold_option in _gold:
                        try:
                            result_score = m.compute(
                                references=[gold_option],
                                predictions=[result],
                            )
                        except TypeError:
                            result_score = m.fn([gold_option, result])
                        if isinstance(result_score, dict):
                            result_score = result_score[metric]
                        scores.append(result_score)
                    result_score = 1.0 if any(scores) else 0.0
            else:
                try:
                    result_score = m.compute(
                        references=[gold],
                        predictions=[result],
                    )
                except TypeError:
                    result_score = m.fn([gold, result])
            if isinstance(result_score, dict):
                for k, v in result_score.items():
                    result_dict[k] = v
            else:
                result_dict[metric] = result_score

        return result_dict


@dataclass
class CustomScorer(Scorer):
    """Scorer that delegates to a user-provided process_results callable."""

    process_results_fn: Callable | None = None

    def _compute_metrics(
        self, instances: list[Instance], filtered_resps: list
    ) -> dict[str, Any]:
        if self.process_results_fn is None:
            raise ValueError("CustomScorer requires a process_results_fn callable.")
        return self.process_results_fn(instances[0].doc, filtered_resps)
