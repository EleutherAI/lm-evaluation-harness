"""Regression tests for the SQuAD2 task.

https://github.com/EleutherAI/lm-evaluation-harness/issues/3212

The task's ``doc_to_target`` teaches the model (via fewshot examples) to emit
the literal string "unanswerable" for unanswerable questions, but the HF
``squad_v2`` metric only credits an *empty* prediction for no-answer
references. ``process_results`` must therefore map an "unanswerable"
generation to an empty ``prediction_text``.
"""

import pytest

from lm_eval.tasks.squadv2.task import SQuAD2, _squad_agg


NO_ANSWER_DOC = {
    "id": "no-answer-0",
    "title": "Normans",
    "context": "The Normans were a population arising in the medieval Duchy of Normandy.",
    "question": "Who gave their name to Normandy in the 1000s and 1100s?",
    "answers": {"text": [], "answer_start": []},
}

HAS_ANSWER_DOC = {
    "id": "has-answer-0",
    "title": "Normans",
    "context": "The Normans were a population arising in the medieval Duchy of Normandy.",
    "question": "Where did the Normans arise?",
    "answers": {"text": ["Normandy"], "answer_start": [63]},
}

# (logprob of " unanswerable", is_greedy) from the loglikelihood request.
LOGLIKELIHOOD_RESULT = (-24.5, False)


@pytest.fixture(scope="module")
def task() -> SQuAD2:
    # process_results/aggregation use no instance state, so bypass __init__
    # (which would download the full dataset).
    return object.__new__(SQuAD2)


def aggregate(task: SQuAD2, doc: dict, continuation: str, keys: list[str]) -> dict:
    results = task.process_results(doc, [continuation, LOGLIKELIHOOD_RESULT])
    return {key: _squad_agg(key, [results[key]]) for key in keys}


@pytest.mark.parametrize(
    "continuation", ["unanswerable", " Unanswerable ", "UNANSWERABLE"]
)
def test_unanswerable_generation_credited_on_no_answer_doc(task, continuation):
    scores = aggregate(task, NO_ANSWER_DOC, continuation, ["NoAns_exact", "NoAns_f1"])
    assert scores["NoAns_exact"] == 100.0
    assert scores["NoAns_f1"] == 100.0


def test_wrong_generation_not_credited_on_no_answer_doc(task):
    scores = aggregate(task, NO_ANSWER_DOC, " Dyrrachium", ["NoAns_exact", "NoAns_f1"])
    assert scores["NoAns_exact"] == 0.0
    assert scores["NoAns_f1"] == 0.0


def test_correct_generation_credited_on_answerable_doc(task):
    scores = aggregate(task, HAS_ANSWER_DOC, " Normandy", ["HasAns_exact", "HasAns_f1"])
    assert scores["HasAns_exact"] == 100.0
    assert scores["HasAns_f1"] == 100.0
