from typing import List

from lm_eval.api.metrics import exact_match_fn


def process_results_mgsm(doc, prediction):
    gold: List = doc["input_correct_responses"]
    return {
        "exact_match": int(
            exact_match_fn(
                predictions=prediction * len(gold), references=gold, ignore_case=True
            )["exact_match"]
            > 0
        )
    }
