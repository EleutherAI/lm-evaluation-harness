# Task-formation variants for nortruthfulqa_mc (Bokmål).
#
# Three multiple-choice formations share each prompt template (p0-p4):
#   - cf     (cloze):    no options in the prompt; the full option text is the
#                        continuation      -> doc_to_choice: mc1_targets.choices
#   - hybrid:            options shown in the prompt WITHOUT labels; the full
#                        option text is the continuation
#                        -> doc_to_choice: mc1_targets.choices
#   - mcf    (mult.-ch): options shown in the prompt WITH labels; only the
#                        label is the continuation
#                        -> doc_to_choice: !function utils.mcf_labels_nb
#
# The dataset has no option letters, so labels (A, B, C, ...) are generated
# positionally from the number of choices.  The options themselves are
# shuffled per document by process_docs in ../utils.py (the dataset lists
# the correct answer first).

import string


def _labels(doc):
    n = len(doc["mc1_targets"]["choices"])
    return list(string.ascii_uppercase[:n])


def _unlabeled_choices(doc):
    """Options as an unlabeled bullet list (used by hybrid formations)."""
    return "".join(f"\n- {choice}" for choice in doc["mc1_targets"]["choices"])


def _labeled_choices(doc):
    """Options as a labeled list, e.g. 'A. ...' (used by mcf formations)."""
    return "".join(
        f"\n {label}. {choice}"
        for label, choice in zip(
            _labels(doc), doc["mc1_targets"]["choices"], strict=True
        )
    )


def mcf_labels_nb(doc):
    """doc_to_choice for the mcf formations: the option labels only."""
    return _labels(doc)


# --- p0: "Spørsmål: {question}" / "Svar:" -----------------------------------
def cf_p0_nb(doc):
    return "Spørsmål: {question}\n\nSvar:".format(question=doc["question"])


def hybrid_p0_nb(doc):
    return "Spørsmål: {question}\n\nAlternativer:{choices}\n\nSvar:".format(
        question=doc["question"], choices=_unlabeled_choices(doc)
    )


def mcf_p0_nb(doc):
    return "Spørsmål: {question}\n\nAlternativer:{choices}\n\nSvar:".format(
        question=doc["question"], choices=_labeled_choices(doc)
    )


# --- p1: "Svaralternativer:" ------------------------------------------------
def cf_p1_nb(doc):
    return "Spørsmål: {question}\n\nSvar:".format(question=doc["question"])


def hybrid_p1_nb(doc):
    return "Spørsmål: {question}\n\nSvaralternativer:{choices}\n\nSvar:".format(
        question=doc["question"], choices=_unlabeled_choices(doc)
    )


def mcf_p1_nb(doc):
    return "Spørsmål: {question}\n\nSvaralternativer:{choices}\n\nSvar:".format(
        question=doc["question"], choices=_labeled_choices(doc)
    )


# --- p2: "Hvilke av følgende alternativer er riktig svar på spørsmålet?" -----
def cf_p2_nb(doc):
    return "Spørsmål: {question}\n\nHva er riktig svar på spørsmålet?\n\nSvar:".format(
        question=doc["question"]
    )


def hybrid_p2_nb(doc):
    return "Spørsmål: {question}\n\nHvilke av følgende alternativer er riktig svar på spørsmålet?{choices}\n\nSvar:".format(
        question=doc["question"], choices=_unlabeled_choices(doc)
    )


def mcf_p2_nb(doc):
    return "Spørsmål: {question}\n\nHvilke av følgende alternativer er riktig svar på spørsmålet?{choices}\n\nSvar:".format(
        question=doc["question"], choices=_labeled_choices(doc)
    )


# --- p3: "Gitt følgende spørsmål, hvilket av de mulige svarene under ..." ----
def cf_p3_nb(doc):
    return "Gitt følgende spørsmål, hva er det riktige svaret?\nSpørsmål: {question}\n\nSvar:".format(
        question=doc["question"]
    )


def hybrid_p3_nb(doc):
    return "Gitt følgende spørsmål, hvilket av de mulige svarene under er riktig?\nSpørsmål: {question}\n{choices}\n\nSvar:".format(
        question=doc["question"], choices=_unlabeled_choices(doc)
    )


def mcf_p3_nb(doc):
    return "Gitt følgende spørsmål, hvilket av de mulige svarene under er riktig?\nSpørsmål: {question}\n{choices}\n\nSvar:".format(
        question=doc["question"], choices=_labeled_choices(doc)
    )


# --- p4: "Velg et av følgende mulige svar:" ---------------------------------
def cf_p4_nb(doc):
    return "{question}\n\nSvar:".format(question=doc["question"])


def hybrid_p4_nb(doc):
    return "{question}\nVelg et av følgende mulige svar:{choices}\n\nSvar:".format(
        question=doc["question"], choices=_unlabeled_choices(doc)
    )


def mcf_p4_nb(doc):
    return "{question}\nVelg et av følgende mulige svar:{choices}\n\nSvar:".format(
        question=doc["question"], choices=_labeled_choices(doc)
    )


# --- accuracy / probability metrics (see noreval_metrics.py) ---------------
from lm_eval.tasks.noreval.noreval_metrics import make_process_results


def choices_text(doc):
    return doc["mc1_targets"]["choices"]


# process_docs (../utils.py) shuffles the choices, so the single correct
# answer is wherever label 1 landed rather than always index 0
def gold(doc):
    return [doc["mc1_targets"]["labels"].index(1)]


process_results = make_process_results(choices_text, gold)  # cf / hybrid
process_results_mcf = make_process_results(mcf_labels_nb, gold)
