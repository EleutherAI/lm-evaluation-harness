"""Scoring for the WriteToLevel lm-evaluation-harness task.

Mirrors benchmark/evaluate.py:score_one. A passage hits its target when at
least 95% of its characters come from the target level or below, graded by
hsk30 against the 2025 HSK examination syllabus. Off-scale text counts as one
step beyond the 7-9 band.
"""

import hsk30

OFF_SCALE = hsk30.BAND + 1


def process_results(doc, results):
    text = str(results[0] or "")
    threshold = doc.get("threshold", 0.95)
    target = int(doc["target_level"])
    profile = hsk30.grade(text, threshold=threshold)
    measured = profile.level if profile.level is not None else OFF_SCALE
    _share, offenders = hsk30.budget_violations(text, target, threshold=threshold)
    return {
        "level_accuracy": float(profile.level is not None and profile.level <= target),
        "signed_error": float(measured - target),
        "length_compliance": float(abs(profile.chars - int(doc["target_chars"])) <= int(doc.get("char_tolerance", 0))),
        "budget_violation": float(any(sh > (1 - threshold) for _c, _l, sh in offenders)),
    }
