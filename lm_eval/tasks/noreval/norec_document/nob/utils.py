from lm_eval.tasks.noreval.noreval_metrics import make_process_results


def choices_negativ_positiv(doc):
    return ["negativ", "positiv"]


def choices_darlig_bra(doc):
    return ["dårlig", "bra"]


def choices_letters(doc):
    return ["A", "B"]


def gold(doc):
    return [doc["sentiment"]]


def _add_f1_pairs(metrics, predictions, doc):
    # (prediction, gold) pairs, aggregated by the macro-F1 in ../utils.py
    for suffix, pred in predictions.items():
        metrics[f"f1{suffix}"] = (pred, doc["sentiment"])
        metrics[f"f1{suffix}_stderr"] = (pred, doc["sentiment"])


process_results = make_process_results(choices_negativ_positiv, gold, _add_f1_pairs)
process_results_darlig_bra = make_process_results(
    choices_darlig_bra, gold, _add_f1_pairs
)
process_results_mcf = make_process_results(choices_letters, gold, _add_f1_pairs)
