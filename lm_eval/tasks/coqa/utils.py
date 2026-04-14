from datasets import Dataset

import transformers.data.metrics.squad_metrics as squad_metrics


def process_docs(dataset):
    """
    Expand each CoQA conversation into multiple instances, one per turn.
    Each instance contains the story and conversation history up to that turn.
    """

    def _expand_turns(doc):
        """Expand a single document into multiple turns."""
        story = doc["story"]
        questions = doc["questions"]["input_text"]
        answers = doc["answers"]["input_text"]
        additional_answers = doc.get("additional_answers", {})

        # Create lists to store all turns
        expanded = {
            "story": [],
            "questions": [],
            "answers": [],
            "additional_answers": [],
            "turn_id": [],
        }

        # Create one instance per turn
        for turn_idx in range(len(questions)):
            expanded["story"].append(story)
            # Store questions and answers up to and including this turn
            expanded["questions"].append(questions[: turn_idx + 1])
            expanded["answers"].append(answers[: turn_idx + 1])
            expanded["turn_id"].append(turn_idx)

            # Handle additional answers for this turn
            turn_additional = {}
            if additional_answers:
                for key, value in additional_answers.items():
                    if "input_text" in value:
                        turn_additional[key] = value["input_text"][turn_idx]
            expanded["additional_answers"].append(turn_additional)

        return expanded

    # Apply the expansion
    dataset = dataset.map(
        _expand_turns,
        remove_columns=[
            key for key in dataset.features.keys() if key not in ["story"]
        ],
    )

    # Flatten the lists
    new_dataset = {}
    for key in dataset.features.keys():
        new_dataset[key] = [x for row in dataset[key] for x in row]

    return Dataset.from_dict(new_dataset)


def doc_to_text(doc):
    # Given a passage p, the conversation history {q1, a1, . . . qi−1, ai−1}
    # and a question qi, the task is to predict the answer ai
    doc_text = doc["story"] + "\n\n"

    questions = doc["questions"]
    answers = doc["answers"]

    # Add conversation history (all Q&A pairs except the last answer)
    for i in range(len(questions) - 1):
        doc_text += f"Q: {questions[i]}\n\n"
        doc_text += f"A: {answers[i]}\n\n"

    # Add the current question without its answer
    doc_text += f"Q: {questions[-1]}\n\nA:"

    return doc_text


def doc_to_target(doc):
    # Returns unique answers and valid alternatives (Some questions in CoQA have multiple valid answers).
    answers = []
    # The target is the last answer in this turn's history
    answer_for_turn = doc["answers"][-1]
    answers.append(answer_for_turn)

    additional_answers = doc.get("additional_answers", {})
    if additional_answers:
        for key, value in additional_answers.items():
            if value and value.lower() not in map(str.lower, answers):
                answers.append(value)

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
