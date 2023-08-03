import numpy as np
import transformers.data.metrics.squad_metrics as squad_metrics

from lm_eval.api.metrics import metric_max_over_ground_truths


def doc_to_text(doc):
    initial_text, *highlights = doc["passage"].strip().split("\n@highlight\n")
    text = initial_text + "\n\n"
    for highlight in highlights:
        text += f"  - {highlight}.\n"
    return text


def format_answer(query, entity):
    return f"  - {query}".replace("@placeholder", entity)


def doc_to_target(doc):
    # We only output the first correct entity in a doc
    return format_answer(query=doc["query"], entity=doc["answers"][0])


def process_results(doc, results):
    # ReCoRD's evaluation is actually deceptively simple:
    # - Pick the maximum likelihood prediction entity
    # - Evaluate the accuracy and token F1 PER EXAMPLE
    # - Average over all examples
    max_idx = np.argmax(np.array([result[0] for result in results]))

    prediction = doc["entities"][max_idx]
    gold_label_set = doc["answers"]
    f1 = metric_max_over_ground_truths(
        squad_metrics.compute_f1, prediction, gold_label_set
    )
    em = metric_max_over_ground_truths(
        squad_metrics.compute_exact, prediction, gold_label_set
    )

    return {
        "f1": f1,
        "em": em,
    }
