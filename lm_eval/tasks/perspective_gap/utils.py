import re


def _strip_think_tags(text):
    """Extract the answer after </think>, or return text as-is."""
    parts = text.rsplit("</think>", 1)
    if len(parts) == 2:
        return parts[1].strip()
    return text.strip()


def process_results_role_assignment(doc, results):
    from perspective_gap.scoring import score_role_assignment

    pred = _strip_think_tags(results[0])
    result = score_role_assignment(
        pred,
        doc["reference_need_sets"],
        doc.get("distractor_id"),
    )
    return result["metrics"]


def process_results_prompt_writing(doc, results):
    from perspective_gap.scoring import score_prompt_writing

    pred = _strip_think_tags(results[0])
    result = score_prompt_writing(
        pred,
        doc["fragments"],
        doc["reference_need_sets"],
        doc.get("distractor_id"),
    )
    return result["metrics"]
