from itertools import zip_longest

import transformers.data.metrics.squad_metrics as squad_metrics


def process_docs(dataset):
    """Expand each story into one document per conversation turn.

    The official CoQA evaluation (https://stanfordnlp.github.io/coqa/)
    scores every turn, not just the last one: for turn ``t`` the model sees
    the passage plus questions/answers 0..t-1 and must produce answer ``t``.
    Each expanded doc carries its 0-based ``turn_id``; doc_to_text and
    doc_to_target honor it and fall back to the last turn when absent.
    """
    docs = []
    for doc in dataset:
        n_turns = len(doc["questions"]["input_text"])
        if n_turns == 0:
            docs.append(doc)
            continue
        for turn_id in range(n_turns):
            expanded = dict(doc)
            expanded["turn_id"] = turn_id
            docs.append(expanded)
    return docs


def doc_to_text(doc):
    # Given a passage p, the conversation history {q1, a1, . . . qi−1, ai−1}
    # and a question qi, the task is to predict the answer ai.
    # `turn_id` selects which turn is queried (default: last turn).
    turn_id = doc.get("turn_id", len(doc["questions"]["input_text"]) - 1)
    doc_text = doc["story"] + "\n\n"
    for q, a in zip_longest(
        doc["questions"]["input_text"][: turn_id + 1],
        doc["answers"]["input_text"][:turn_id],
    ):  # omit target answer ai
        question = f"Q: {q}\n\n"
        answer = f"A: {a}\n\n" if a is not None else "A:"
        doc_text += question + answer
    return doc_text


def doc_to_target(doc):
    # Returns unique answers and valid alternatives (Some questions in CoQA have multiple valid answers).
    # `turn_id` selects which turn is targeted (default: last turn).
    turn_id = doc.get("turn_id", len(doc["questions"]["input_text"]) - 1)
    answers = []
    answer_forturn = doc["answers"]["input_text"][turn_id]
    answers.append(answer_forturn)

    additional_answers = doc.get("additional_answers")
    if additional_answers:
        for key in additional_answers:
            additional_answer_for_turn = additional_answers[key]["input_text"][
                turn_id
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
