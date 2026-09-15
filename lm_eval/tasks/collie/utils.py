"""Helpers for the `collie` task."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from lm_eval.tasks.collie.constraints import (
    All,
    Constraint,
    Count,
    ForEach,
    InputLevel,
    Position,
    Reduction,
    Relation,
    TargetLevel,
)


if TYPE_CHECKING:
    from typing import Any

    from lm_eval.tasks.collie.constraints import LevelStr, OperandStr, ReductionStr


def _make_constraint(
    target_level: LevelStr,
    transformation: Any,
    operand: OperandStr,
    input_level: LevelStr | None = None,
    reduction: ReductionStr | None = None,
) -> Constraint:
    return Constraint(
        input_level=InputLevel(input_level),
        target_level=TargetLevel(target_level),
        transformation=transformation,
        relation=Relation(operand),
        reduction=Reduction(reduction),
    )


# The 13 COLLIE-v1 constraint structures, reconstructed from the objects in the official `all_data.dill`.
CONSTRAINTS: dict[str, Constraint | All] = {
    # word level
    "c01": _make_constraint("character", Count(), ">="),
    "c02": All(
        _make_constraint("character", Count(), "=="),
        _make_constraint("character", Position([3, 7, 10]), "=="),
    ),
    "c03": All(
        _make_constraint("character", Count(), "=="),
        _make_constraint("character", Position(-1), "=="),
    ),
    # sentence level
    "c04": _make_constraint("character", Count(), "=="),
    "c05": All(
        _make_constraint("word", Count(), "=="),
        _make_constraint("word", Position([3, 7, 10]), "=="),
    ),
    "c06a": All(
        _make_constraint("word", Count(), ">="),
        _make_constraint(
            "character", ForEach(Count()), "<=", input_level="word", reduction="all"
        ),
    ),
    "c07": _make_constraint("word", ForEach(...), "in"),
    # paragraph level
    "c08": _make_constraint(
        "word", ForEach(Position(0)), "==", input_level="sentence", reduction="all"
    ),
    "c09": All(
        _make_constraint("sentence", Count(), ">="),
        _make_constraint("word", ForEach(...), "not in"),
        _make_constraint("word", ForEach(...), "not in"),
        _make_constraint("word", ForEach(...), "not in"),
    ),
    "c10": All(
        _make_constraint("sentence", Count(), "=="),
        _make_constraint(
            "word", ForEach(Count()), ">=", input_level="sentence", reduction="all"
        ),
        _make_constraint(
            "word", ForEach(Count()), "<=", input_level="sentence", reduction="all"
        ),
    ),
    "c11": All(
        _make_constraint("sentence", Count(), ">="),
        _make_constraint(
            "word", ForEach(Count()), ">=", input_level="sentence", reduction="all"
        ),
    ),
    "c12": All(
        _make_constraint("sentence", Count(), "=="),
        _make_constraint(
            "word", ForEach(Position(-1)), "==", input_level="sentence", reduction="all"
        ),
    ),
    # passage level
    "c14": _make_constraint(
        "sentence",
        ForEach(Position(-1)),
        "==",
        input_level="paragraph",
        reduction="all",
    ),
}


def process_results(doc: dict[str, Any], results: list[str]) -> dict[str, bool]:
    """Score a generation against its COLLIE constraint (pass-rate accuracy)."""
    constraint = CONSTRAINTS[doc["constraint_id"]]
    targets = json.loads(doc["targets"])
    try:
        passed = bool(constraint.check(results[0], targets))
    except Exception:  # noqa: BLE001
        passed = False
    return {"acc": passed}
