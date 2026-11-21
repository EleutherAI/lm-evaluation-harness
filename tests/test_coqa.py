import types

import datasets

from lm_eval.tasks.coqa import utils as coqa_utils


def _make_doc():
    # A single CoQA-style story with three conversation turns.
    return {
        "id": "s1",
        "source": "test",
        "story": "STORY",
        "questions": {
            "input_text": ["q1", "q2", "q3"],
            "turn_id": [1, 2, 3],
        },
        "answers": {
            "input_text": ["a1", "a2", "a3"],
            "turn_id": [1, 2, 3],
        },
        "additional_answers": {
            "0": {"input_text": ["a1", "a2alt", "a3"]},
            "1": {"input_text": ["a1", "a2", "a3"]},
            "2": {"input_text": ["a1", "A2ALT", "a3"]},
        },
    }


def test_process_docs_expands_one_doc_per_turn():
    dataset = datasets.Dataset.from_list([_make_doc()])

    expanded = coqa_utils.process_docs(dataset)

    # The three-turn story must become three scored instances.
    assert len(expanded) == 3
    assert [row["turn_id"] for row in expanded] == [1, 2, 3]
    # Every expanded row keeps the full story/question/answer context.
    for row in expanded:
        assert row["story"] == "STORY"
        assert row["questions"]["input_text"] == ["q1", "q2", "q3"]
        assert row["answers"]["input_text"] == ["a1", "a2", "a3"]


def test_doc_to_text_includes_history_up_to_current_turn():
    doc = {**_make_doc(), "turn_id": 2}

    text = coqa_utils.doc_to_text(doc)

    # Context holds the prior turn's answer but stops before the target answer.
    assert text == "STORY\n\nQ: q1\n\nA: a1\n\nQ: q2\n\nA:"


def test_doc_to_text_first_turn_has_no_history():
    doc = {**_make_doc(), "turn_id": 1}

    assert coqa_utils.doc_to_text(doc) == "STORY\n\nQ: q1\n\nA:"


def test_doc_to_target_returns_answer_for_requested_turn():
    doc = {**_make_doc(), "turn_id": 2}

    targets = coqa_utils.doc_to_target(doc)

    # Primary answer first, then every additional answer, duplicates included.
    assert targets == ["a2", "a2alt", "a2", "A2ALT"]


def test_duplicate_references_score_like_official_evaluator(monkeypatch):
    # transformers>=5 dropped squad_metrics; exact-match stand-ins isolate the averaging.
    def exact(gold, pred):
        return int(gold == pred)

    monkeypatch.setattr(
        coqa_utils,
        "_squad_metrics",
        lambda: types.SimpleNamespace(compute_exact=exact, compute_f1=exact),
    )
    # CoQA dev story 1, turn 4: one reference differs, three annotators agree.
    gold = coqa_utils.doc_to_target(
        {
            "story": "STORY",
            "questions": {"input_text": ["q"]},
            "answers": {"input_text": ["with her mommy and 5 sisters"]},
            "additional_answers": {
                "0": {"input_text": ["her mommy and 5 other sisters"]},
                "1": {"input_text": ["her mommy and 5 other sisters"]},
                "2": {"input_text": ["her mommy and 5 other sisters"]},
            },
        }
    )

    scores = coqa_utils.compute_scores(gold, "her mommy and 5 other sisters")

    # Every leave-one-out subset still contains a matching reference.
    assert len(gold) == 4
    assert scores == {"em": 1.0, "f1": 1.0}


def test_helpers_fall_back_to_last_turn_without_turn_id():
    doc = _make_doc()  # no explicit turn_id -> behave as last turn

    assert coqa_utils.doc_to_text(doc).endswith("Q: q3\n\nA:")
    assert coqa_utils.doc_to_target(doc)[0] == "a3"
