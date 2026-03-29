import logging

import datasets


# Adapted from lm_eval/tasks/minerva_math/utils.py [1].
# Key differences: evaluation relies exclusively on math_verify's symbolic parsing,
# dropping the exact-match fallback (is_equiv) used in minerva_math, which is insufficient
# for the algebraic complexity of PolyMath's high and top difficulty tiers.
# aggregate_dw_acc is original to this task.
# [1] https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/minerva_math/utils.py

eval_logger = logging.getLogger(__name__)

try:
    from importlib.metadata import version as get_version

    from math_verify import parse, verify
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

    assert get_version("antlr4-python3-runtime").startswith("4.11")
except (ModuleNotFoundError, AssertionError) as e:
    raise type(e)(
        "`math_verify` and `antlr4-python3-runtime==4.11` are required because the evaluation logic "
        "relies on symbolic parsing to verify mathematical equivalence. "
        "Please install the required packages via pip install lm-eval[math] or pip install -e .[math]"
    ) from e


def aggregate_dw_acc(answers: dict[str, float]) -> float:
    """
    Compute the Difficulty-Weighted Accuracy (DW-ACC). See section "2.5 Benchmark Score: Difficulty-Weighted Accuracy"
        in the PolyMAth paper (https://arxiv.org/pdf/2504.18428).

    Args:
        answers: Dictionary mapping task names to their accuracy scores. The task name must contain a difficulty
            keyword: 'low', 'medium', 'high', or 'top'.

    Returns:
        float: Difficulty-weighted average accuracy across all tasks.
    """
    weights = {"low": 1, "medium": 2, "high": 4, "top": 8}
    weighted_sum = 0
    total_weight = 0
    for task_name, score in answers.items():
        level = next((l for l in weights if l in task_name.lower()), None)
        if level:
            w = weights[level]
            weighted_sum += score * w
            total_weight += w
    return weighted_sum / total_weight if total_weight > 0 else 0


# Diverges from minerva_math:
# https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/minerva_math/utils.py
# Uses math_verify symbolic parsing only (no exact_match fallback via is_equiv/normalize_final_answer),
# and reads from doc["answer"] directly instead of doc["solution"].
# \boxed{} extraction is handled natively by parse() via LatexExtractionConfig(boxed_match_priority=0),
# making manual extraction (e.g. remove_boxed, last_boxed_only_string) unnecessary.
def process_results(doc: dict, results: list[str]) -> dict[str, int]:
    _mvres = verify(
        gold=parse(doc["answer"]),
        target=parse(
            results[0],
            extraction_config=[
                LatexExtractionConfig(boxed_match_priority=0),
                ExprExtractionConfig(),
            ],
        ),
    )
    return {"math_verify": 1 if _mvres else 0}


# Diverges from minerva_math:
# https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/minerva_math/utils.py
# Uses raw "question"/"answer" fields directly; no normalize_final_answer, remove_boxed,
# or few_shot handling, as the PolyMath dataset provides pre-cleaned answers.
def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        out_doc = {
            "question": doc["question"],
            "answer": doc["answer"],
        }
        return out_doc

    return dataset.map(_process_doc)
