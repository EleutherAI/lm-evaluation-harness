from lm_eval.tasks.tomato import utils as tomato_utils


def test_load_recreates_original_label_space():
    doc = tomato_utils.load()["test"][0]

    assert doc["choice_labels"] == ["[A]", "[B]", "[C]", "[D]"]
    assert doc["gold_label"] in doc["choice_labels"]


def test_partition_row_counts_match_original_summary():
    dataset = tomato_utils.load()["test"]

    assert len(dataset) == 5401
    assert len(tomato_utils.process_docs_order_1(dataset)) == 2676
    assert len(tomato_utils.process_docs_order_2(dataset)) == 2725
    assert len(tomato_utils.process_docs_mental_state_belief(dataset)) == 1109
    assert len(tomato_utils.process_docs_mental_state_desire(dataset)) == 1025
    assert len(tomato_utils.process_docs_mental_state_emotion(dataset)) == 996
    assert len(tomato_utils.process_docs_mental_state_intention(dataset)) == 1110
    assert len(tomato_utils.process_docs_mental_state_knowledge(dataset)) == 1161
    assert len(tomato_utils.process_docs_false_belief_false(dataset)) == 4595
    assert len(tomato_utils.process_docs_false_belief_true(dataset)) == 806


def test_doc_to_text_matches_original_prompt_shape():
    doc = tomato_utils.load()["test"][0]
    prompt = tomato_utils.doc_to_text(doc)

    assert prompt.startswith("# Transcript \n")
    assert "\n\n# Question \n" in prompt
    assert "\n\n# Options \n[A] " in prompt
    assert "Answer:" not in prompt
