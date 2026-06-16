from itertools import zip_longest

from datasets import Dataset
from transformers.data.metrics import squad_metrics


def _slice_mapping_values(mapping, stop):
    return {
        key: value[:stop] if isinstance(value, list) else value
        for key, value in mapping.items()
    }


def _doc_for_turn(doc, turn_idx):
    stop = turn_idx + 1
    turn_doc = {
        "story": doc["story"],
        "questions": _slice_mapping_values(doc["questions"], stop),
        "answers": _slice_mapping_values(doc["answers"], stop),
    }

    for key in ("id", "source", "filename", "name"):
        if key in doc:
            turn_doc[key] = doc[key]

    additional_answers = doc.get("additional_answers")
    if additional_answers:
        turn_doc["additional_answers"] = {
            key: _slice_mapping_values(answers, stop)
            for key, answers in additional_answers.items()
        }

    return turn_doc


def process_docs(dataset):
    turn_docs = []
    for doc in dataset:
        num_turns = len(doc["questions"]["input_text"])
        turn_docs.extend(_doc_for_turn(doc, turn_idx) for turn_idx in range(num_turns))
    return Dataset.from_list(turn_docs)


def doc_to_text(doc):
    # Given a passage p, the conversation history {q1, a1, . . . qi−1, ai−1}
    # and a question qi, the task is to predict the answer ai
    doc_text = doc["story"] + "\n\n"
    for q, a in zip_longest(
        doc["questions"]["input_text"], doc["answers"]["input_text"][:-1]
    ):  # omit target answer ai
        question = f"Q: {q}\n\n"
        answer = f"A: {a}\n\n" if a is not None else "A:"
        doc_text += question + answer
    return doc_text


def doc_to_target(doc):
    turn_id = len(doc["questions"]["input_text"])
    # Returns unique answers and valid alternatives (Some questions in CoQA have multiple valid answers).
    answers = []
    answer_forturn = doc["answers"]["input_text"][turn_id - 1]
    answers.append(answer_forturn)

    additional_answers = doc.get("additional_answers")
    if additional_answers:
        for key in additional_answers:
            additional_answer_for_turn = additional_answers[key]["input_text"][
                turn_id - 1
            ]
            if additional_answer_for_turn.lower() not in map(str.lower, answers):
                answers.append(additional_answer_for_turn)
    return answers


def em(gold_list, pred):
    # tests for exact match and on the normalised answer (compute_exact)
    em_sum = 0.0
    if len(gold_list) > 1:
        for i in range(len(gold_list)):
            gold_answers = gold_list[0:i] + gold_list[i + 1 :]
            # predictions compared against (n) golds and take maximum
            em_sum += max(squad_metrics.compute_exact(a, pred) for a in gold_answers)
    else:
        em_sum += max(squad_metrics.compute_exact(a, pred) for a in gold_list)

    return em_sum / max(1, len(gold_list))


def compute_scores(gold_list, pred):
    # tests for exact match and on the normalised answer (compute_exact)
    # test for overlap (compute_f1)
    f1_sum = 0.0
    em_sum = 0.0
    if len(gold_list) > 1:
        for i in range(len(gold_list)):
            gold_answers = gold_list[0:i] + gold_list[i + 1 :]
            # predictions compared against (n) golds and take maximum
            em_sum += max(squad_metrics.compute_exact(a, pred) for a in gold_answers)
            f1_sum += max(squad_metrics.compute_f1(a, pred) for a in gold_answers)
    else:
        em_sum += max(squad_metrics.compute_exact(a, pred) for a in gold_list)
        f1_sum += max(squad_metrics.compute_f1(a, pred) for a in gold_list)

    return {
        "em": em_sum / max(1, len(gold_list)),
        "f1": f1_sum / max(1, len(gold_list)),
    }


def process_results(doc, results):
    gold_list = doc_to_target(doc)
    pred = results[0].strip().split("\n")[0]

    scores = compute_scores(gold_list, pred)
    return scores
