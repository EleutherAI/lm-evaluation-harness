from lm_eval.tasks.noreval.noreval_metrics import make_process_results


def choices_text(doc):
    return doc["choices"]["text"]


def choices_labels(doc):
    return doc["choices"]["label"]


def gold(doc):
    return [doc["choices"]["label"].index(doc["answer"])]


process_results = make_process_results(choices_text, gold)  # cf / hybrid
process_results_mcf = make_process_results(choices_labels, gold)
