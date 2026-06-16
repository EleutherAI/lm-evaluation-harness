from datasets import Dataset

from lm_eval.tasks.coqa import utils


def _sample_doc():
    return {
        "id": "story-1",
        "story": "Mary went to the shop.",
        "questions": {
            "input_text": ["Where did Mary go?", "Why?"],
            "turn_id": [1, 2],
        },
        "answers": {
            "input_text": ["the shop", "to buy milk"],
            "turn_id": [1, 2],
        },
        "additional_answers": {
            "turk1": {
                "input_text": ["a shop", "for milk"],
                "turn_id": [1, 2],
            },
            "turk2": {
                "input_text": ["The shop", "to buy milk"],
                "turn_id": [1, 2],
            },
        },
    }


def test_process_docs_expands_each_coqa_turn():
    processed = utils.process_docs(Dataset.from_list([_sample_doc()]))

    assert len(processed) == 2
    assert processed[0]["questions"]["input_text"] == ["Where did Mary go?"]
    assert processed[0]["answers"]["input_text"] == ["the shop"]
    assert processed[0]["additional_answers"]["turk1"]["input_text"] == ["a shop"]
    assert processed[1]["questions"]["input_text"] == [
        "Where did Mary go?",
        "Why?",
    ]
    assert processed[1]["answers"]["input_text"] == ["the shop", "to buy milk"]


def test_all_turn_docs_use_current_turn_as_target():
    processed = utils.process_docs(Dataset.from_list([_sample_doc()]))

    assert utils.doc_to_text(processed[0]) == (
        "Mary went to the shop.\n\nQ: Where did Mary go?\n\nA:"
    )
    assert utils.doc_to_target(processed[0]) == ["the shop", "a shop"]
    assert utils.doc_to_text(processed[1]) == (
        "Mary went to the shop.\n\n"
        "Q: Where did Mary go?\n\n"
        "A: the shop\n\n"
        "Q: Why?\n\n"
        "A:"
    )
    assert utils.doc_to_target(processed[1]) == ["to buy milk", "for milk"]


def test_unsplit_docs_keep_last_turn_behaviour():
    assert utils.doc_to_target(_sample_doc()) == ["to buy milk", "for milk"]
