from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from lm_eval.api._types import Reference


@dataclass(slots=True)
class ScoredDoc:
    """Per-document scoring result produced by a Scorer.

    Bundles the document reference and all metric scores together so that
    downstream reduction never needs to align parallel lists.
    """

    doc_id: int
    reference: Reference
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
