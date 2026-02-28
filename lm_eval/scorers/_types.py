from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias


if TYPE_CHECKING:
    from lm_eval.api._types import Reference


@dataclass(frozen=True, slots=True)
class ScoredDoc:
    """Immutable per-document raw scoring result.

    Created by ``score_doc()`` / ``score_instances()``.  Contains per-repeat
    values that haven't been reduced yet.  After reduction, a
    :class:`ReducedDoc` is produced — ``ScoredDoc`` itself is never mutated.
    """

    doc_id: int
    reference: Reference
    scores: dict[str, list[float]]  # {metric_name: [per_repeat_values]}


ReducedDoc: TypeAlias = dict[str, float]
"""Per-document reduced result: ``{metric_name: scalar_value}``.

The doc_id is the key in the containing ``dict[int, ReducedDoc]``.
Created by :meth:`Scorer.reduce` from a :class:`ScoredDoc`, or directly
by :meth:`Scorer.import_reduced` after a distributed gather.
"""


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
