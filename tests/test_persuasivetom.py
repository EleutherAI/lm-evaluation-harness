from lm_eval.tasks.persuasivetom import utils as persuasivetom_utils


def test_load_keeps_malformed_intent_er_rows_in_denominator():
    dataset = persuasivetom_utils.load(data_file="intent_er.json")["test"]

    assert len(dataset) == 2568
    malformed = [
        row for row in dataset if row["answerKey"] not in row["choice_letters"]
    ]
    assert len(malformed) == 20


def test_doc_to_text_matches_original_user_prompt_shape():
    doc = persuasivetom_utils.load(data_file="desire_er.json")["test"][0]
    prompt = persuasivetom_utils.doc_to_text(doc)

    assert prompt.startswith("\nDialogue History:\n")
    assert "\nQuestion:\n" in prompt
    assert prompt.endswith("\nAnswer:")


def test_process_results_scores_malformed_gold_wrong():
    doc = {
        "choice_letters": ["A", "B", "C", "D", "E", "F"],
        "answerKey": "None of the above",
    }

    assert persuasivetom_utils.process_results(doc, ["Answer: A"]) == {"acc": 0.0}
