# Task-formation variants for nrk_quiz_qa (Bokmål).
#
# Three multiple-choice formations share each prompt template (p0-p4):
#   - cf     (cloze):    no options in the prompt; the full option text is the
#                        continuation      -> doc_to_choice: choices.text
#   - hybrid:            options shown in the prompt WITHOUT labels; the full
#                        option text is the continuation
#                        -> doc_to_choice: choices.text
#   - mcf    (mult.-ch): options shown in the prompt WITH labels; only the
#                        label is the continuation
#                        -> doc_to_choice: choices.label


def _unlabeled_choices(doc):
    """Options as an unlabeled bullet list (used by hybrid formations)."""
    return "".join(f"\n- {text}" for text in doc["choices"]["text"])


def _labeled_choices(doc):
    """Options as a labeled list, e.g. 'A. Oslo' (used by mcf formations)."""
    return "".join(
        f"\n {label}. {option}"
        for label, option in zip(
            doc["choices"]["label"], doc["choices"]["text"], strict=True
        )
    )


def _enumerate(items):
    """Join items as 'a, b eller c' (or 'a eller b' for two items)."""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} eller {items[1]}"
    return ", ".join(items[:-1]) + f" eller {items[-1]}"


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


# --- p1: "Svaralternativer:" ... "Hva er riktig svar?" ----------------------
def cf_p1_nb(doc):
    return "{question}\n\nHva er riktig svar?\n\nSvar:".format(question=doc["question"])


def hybrid_p1_nb(doc):
    return "{question}\n\nSvaralternativer:{choices}\n\nHva er riktig svar?\n\nSvar:".format(
        question=doc["question"], choices=_unlabeled_choices(doc)
    )


def mcf_p1_nb(doc):
    return "{question}\n\nSvaralternativer:{choices}\n\nHva er riktig svar?\n\nSvar:".format(
        question=doc["question"], choices=_labeled_choices(doc)
    )


# --- p2: "Er det riktige svaret {...}?" -------------------------------------
def cf_p2_nb(doc):
    return "{question}\n\nHva er det riktige svaret?\n\nSvar:".format(
        question=doc["question"]
    )


def hybrid_p2_nb(doc):
    return "{question}{choices}\n\nEr det riktige svaret {enumerated}?\n\nSvar:".format(
        question=doc["question"],
        choices=_unlabeled_choices(doc),
        enumerated=_enumerate(doc["choices"]["text"]),
    )


def mcf_p2_nb(doc):
    return "{question}{choices}\n\nEr det riktige svaret {enumerated}?\n\nSvar:".format(
        question=doc["question"],
        choices=_labeled_choices(doc),
        enumerated=_enumerate(doc["choices"]["label"]),
    )


# --- p3: "Spørsmål: {question}{options}" / "Svar:" --------------------------
def cf_p3_nb(doc):
    return "Spørsmål: {question}\n\nSvar:".format(question=doc["question"])


def hybrid_p3_nb(doc):
    return "Spørsmål: {question}{choices}\n\nSvar:".format(
        question=doc["question"], choices=_unlabeled_choices(doc)
    )


def mcf_p3_nb(doc):
    return "Spørsmål: {question}{choices}\n\nSvar:".format(
        question=doc["question"], choices=_labeled_choices(doc)
    )


# --- p4: "Velg riktig svar blant disse alternativene:" ----------------------
def cf_p4_nb(doc):
    return "{question}\n\nSvar:".format(question=doc["question"])


def hybrid_p4_nb(doc):
    return "{question}\nVelg riktig svar blant disse alternativene:{choices}\n\nSvar:".format(
        question=doc["question"], choices=_unlabeled_choices(doc)
    )


def mcf_p4_nb(doc):
    return "{question}\nVelg riktig svar blant disse alternativene:{choices}\n\nSvar:".format(
        question=doc["question"], choices=_labeled_choices(doc)
    )


# --- accuracy / probability metrics (see noreval_metrics.py) ---------------
from lm_eval.tasks.noreval.noreval_metrics import make_process_results


def choices_text(doc):
    return doc["choices"]["text"]


def choices_labels(doc):
    return doc["choices"]["label"]


def gold(doc):
    return [doc["choices"]["label"].index(doc["answer"])]


process_results = make_process_results(choices_text, gold)  # cf / hybrid
process_results_mcf = make_process_results(choices_labels, gold)
