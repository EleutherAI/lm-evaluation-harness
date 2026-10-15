import re
from typing import Any

import evaluate

from lm_eval.tasks.catalan_bench.utils import (
    process_doc_cabreu,
    process_results_qa as process_results_xquad_ca,
)
from lm_eval.tasks.galician_bench.utils import process_summarization
from lm_eval.tasks.hellaswag.utils import process_docs as process_docs_hellaswag
from lm_eval.tasks.spanish_bench.utils import process_xlsum
from lm_eval.tasks.xquad.utils import process_results_qa as process_results_xquad


def process_results_mgsm(doc: dict[str, Any], results: list[str]) -> dict[str, int]:
    """Return legacy numeric exact match and Math-Verify scores for MGSM."""
    from math_verify import parse, verify

    response = results[0]
    target = str(doc["answer_number"])

    match = re.search(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", response)
    extracted = match.group(0) if match else response
    normalized_extracted = extracted.replace(",", "").replace("$", "").strip()
    normalized_target = target.replace(",", "").replace("$", "").strip()
    exact_match = int(normalized_extracted == normalized_target)

    try:
        math_verify = int(verify(gold=parse(target), target=parse(response)))
    except Exception:
        math_verify = 0

    return {"exact_match": exact_match, "math_verify": math_verify}


def process_xsum(dataset):
    def _process_doc(doc):
        # Remove double spaces
        doc["document"] = re.sub(r" +", " ", doc["document"])
        doc["summary"] = re.sub(r" +", " ", doc["summary"])
        return doc

    return dataset.map(lambda doc: _process_doc(doc))


def rouge1(items):
    """
    # passthrough for efficiency
    """
    return items


def rouge1_agg(items):
    """
    Higher is better
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    rouge_scorer = evaluate.load("rouge")
    return rouge_scorer.compute(predictions=preds, references=refs)["rouge1"]
