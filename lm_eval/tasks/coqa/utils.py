def _squad_metrics():
    # Imported lazily: it pulls in torch, which the scoring step needs but the
    # dataset-processing helpers (doc_to_text / doc_to_target) do not.
    from transformers.data.metrics import squad_metrics

    return squad_metrics


def process_docs(dataset):
    # CoQA is a multi-turn QA dataset: each story has several questions that must
    # all be scored (the authors average over every turn). Expand each story into
    # one doc per turn, tagging it with a 1-indexed ``turn_id``.
    def _expand(batch):
        keys = list(batch.keys())
        expanded = {key: [] for key in keys}
        expanded["turn_id"] = []
        for idx in range(len(batch["story"])):
            n_turns = len(batch["questions"][idx]["input_text"])
            for turn in range(1, n_turns + 1):
                for key in keys:
                    expanded[key].append(batch[key][idx])
                expanded["turn_id"].append(turn)
        return expanded

    return dataset.map(_expand, batched=True, remove_columns=dataset.column_names)


def _target_turn(doc):
    # Prefer the expanded per-turn index; fall back to the last turn so the
    # helpers still work on un-expanded docs.
    return doc.get("turn_id", len(doc["questions"]["input_text"]))


def doc_to_text(doc):
    # Given a passage p, the conversation history {q1, a1, . . . qi−1, ai−1}
    # and a question qi, the task is to predict the answer ai
    turn_id = _target_turn(doc)
    questions = doc["questions"]["input_text"]
    answers = doc["answers"]["input_text"]
    doc_text = doc["story"] + "\n\n"
    for i in range(turn_id):
        doc_text += f"Q: {questions[i]}\n\n"
        doc_text += f"A: {answers[i]}\n\n" if i < turn_id - 1 else "A:"
    return doc_text


def doc_to_target(doc):
    turn_id = _target_turn(doc)
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
    squad_metrics = _squad_metrics()
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
    squad_metrics = _squad_metrics()
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
