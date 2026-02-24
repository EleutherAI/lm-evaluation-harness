"""Scorer subclasses for extraction-based generation tasks.

Each class sets ``default_filter_cfg`` / ``default_metric_cfg`` ClassVars so
that ``Scorer.from_dict`` picks them up automatically — no intermediate
``ExtractionConfig`` step required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from lm_eval.api.registry import register_scorer
from lm_eval.scorers._base import GenScorer


_EXACT_MATCH_METRIC: list[dict[str, Any]] = [
    {
        "metric": "exact_match",
        "aggregation": "mean",
        "higher_is_better": True,
        "kwargs": {"ignore_case": True, "ignore_punctuation": True},
    }
]


@register_scorer("first_token")
@dataclass
class FirstTokenScorer(GenScorer):
    """Scorer that strips whitespace before matching (single-token extraction)."""

    default_filter_cfg: ClassVar[list[dict[str, Any]]] = [
        {"function": "remove_whitespace"},
    ]
    default_metric_cfg: ClassVar[list[dict[str, Any]]] = _EXACT_MATCH_METRIC


@register_scorer("regex")
@dataclass
class RegexExtractionScorer(GenScorer):
    """Scorer that applies regex extraction then takes the first match."""

    default_filter_cfg: ClassVar[list[dict[str, Any]]] = [
        {"function": "regex"},
        {"function": "take_first"},
    ]
    default_metric_cfg: ClassVar[list[dict[str, Any]]] = _EXACT_MATCH_METRIC


@register_scorer("choice_match")
@dataclass
class ChoiceMatchScorer(GenScorer):
    """Scorer for free-form generation scored by exact match against choices."""

    default_metric_cfg: ClassVar[list[dict[str, Any]]] = _EXACT_MATCH_METRIC
