from lm_eval.tasks.afrimedqa import utils_afrimedqa as utils


def test_parse_choices_compacts_valid_options():
    options = {
        "option1": "N/A",
        "option2": "Liver function tests",
        "option3": "Urinalysis",
        "option4": "Endocervical swab",
        "option5": "C-Reactive protein",
    }

    choices, mapping = utils._parse_choices(options)

    assert [choice["label"] for choice in choices] == ["A", "B", "C", "D"]
    assert [choice["source_key"] for choice in choices] == [
        "option2",
        "option3",
        "option4",
        "option5",
    ]
    assert mapping == {
        "option2": "A",
        "option3": "B",
        "option4": "C",
        "option5": "D",
    }


def test_process_doc_keeps_any_correct_gold_labels():
    doc = {
        "question_type": "mcq",
        "split": "test",
        "question": "Concerning HIV which one is true?",
        "question_clean": "Concerning HIV which one is true?",
        "answer_options": '{"option1":"A valid answer","option2":"Another valid answer","option3":"N/A","option4":"Third valid answer","option5":"N/A"}',
        "correct_answer": "option1,option2,option3,option4",
    }

    processed = utils._process_doc(doc)

    assert processed["choice_labels"] == ["A", "B", "C"]
    assert processed["choices"] == [
        "A valid answer",
        "Another valid answer",
        "Third valid answer",
    ]
    assert processed["gold_labels"] == ["A", "B", "C"]


def test_process_doc_drops_invalid_gold_labels_after_compacting():
    doc = {
        "question_type": "mcq",
        "split": "test",
        "question": "Which answer is valid?",
        "question_clean": "Which answer is valid?",
        "answer_options": {
            "option1": "N/A",
            "option2": "First valid answer",
            "option3": "Second valid answer",
            "option4": "N/A",
            "option5": "Third valid answer",
        },
        "correct_answer": "option1,option3,option4,option5",
    }

    processed = utils._process_doc(doc)

    assert processed["choice_labels"] == ["A", "B", "C"]
    assert processed["gold_labels"] == ["B", "C"]
    assert utils.doc_to_target(processed) == "B, C"


def test_process_results_gen_scores_any_correct():
    doc = {"gold_labels": ["A", "C"]}

    assert utils.process_results_gen(doc, ["The correct option is C."]) == {"acc": 1.0}
    assert utils.process_results_gen(doc, ["B"]) == {"acc": 0.0}


def test_extract_mcq_answer():
    assert utils.extract_mcq_answer("B") == "B"
    assert utils.extract_mcq_answer("Answer: c") == "C"
    assert utils.extract_mcq_answer("No valid answer") == ""


def test_evaluable_filter_excludes_invalid_rows():
    assert utils._is_evaluable({"choices": ["A", "B"], "gold_labels": ["A"]})
    assert not utils._is_evaluable({"choices": ["A"], "gold_labels": ["A"]})
    assert not utils._is_evaluable({"choices": ["A", "B"], "gold_labels": []})
