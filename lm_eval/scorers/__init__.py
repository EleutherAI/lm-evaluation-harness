from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lm_eval.scorers.base import Scorer


if TYPE_CHECKING:
    from lm_eval.config.metric import Metric


def build_scorers_from_config(
    filter_list: list[dict[str, Any]] | None,
    global_metrics: list[Metric],
) -> list[Scorer]:
    """Build a list of Scorers from TaskConfig fields.

    Args:
        filter_list: The ``filter_list`` field from a TaskConfig (may be None).
        global_metrics: The task-level metrics to use when a filter entry
            doesn't specify its own ``metric_list``.

    Returns:
        A list of Scorer objects. Falls back to a single default scorer
        (take_first filter) when *filter_list* is None or empty.
    """
    if not filter_list:
        return [Scorer.default_scorer(global_metrics)]

    return [
        Scorer.from_dict(entry, global_metrics=global_metrics) for entry in filter_list
    ]


__all__ = [
    "Scorer",
    "build_scorers_from_config",
]
