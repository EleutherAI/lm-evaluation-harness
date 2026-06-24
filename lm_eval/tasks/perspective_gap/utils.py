import re
import warnings

_SCORER_INSTALL_CMD = (
    'pip install "perspective-gap @ '
    'git+https://github.com/WhymustIhaveaname/PerspectiveGap.git"'
)


def _load_scorer(name):
    try:
        from perspective_gap import scoring
    except ImportError:
        warnings.warn(
            "Evaluating PerspectiveGap requires the optional "
            "`perspective-gap` dependency. Install it with: "
            f"{_SCORER_INSTALL_CMD}",
            RuntimeWarning,
            stacklevel=2,
        )
        raise
    return getattr(scoring, name)


def _strip_think_tags(text):
    """Extract the answer after </think>, or return text as-is."""
    parts = text.rsplit("</think>", 1)
    if len(parts) == 2:
        return parts[1].strip()
    return text.strip()


def process_results_role_assignment(doc, results):
    score_role_assignment = _load_scorer("score_role_assignment")
    pred = _strip_think_tags(results[0])
    result = score_role_assignment(
        pred,
        doc["reference_need_sets"],
        doc.get("distractor_id"),
    )
    return result["metrics"]


def process_results_prompt_writing(doc, results):
    score_prompt_writing = _load_scorer("score_prompt_writing")
    pred = _strip_think_tags(results[0])
    result = score_prompt_writing(
        pred,
        doc["fragments"],
        doc["reference_need_sets"],
        doc.get("distractor_id"),
    )
    return result["metrics"]
