CHOICES = ["A", "B", "C", "D"]
OPTS = ["option_a", "option_b", "option_c", "option_d"]

# Single source of truth: task suffix → dataset subject value
SUBJECT_MAP = {
    "bodo": "Bodo",
    "dogri": "Dogri",
    "gujarati": "Gujarati_surya",
    "konkani": "Konkani",
    "maithili": "Maithili",
    "marathi": "Marathi",
    "nepali": "Nepali",
    "oriya": "Oriya",
    "rajasthani": "Rajasthani",
    "sanskrit": "Sanskrit",
    "sanskrit_mix": "Sanskrit Mix",
    "santali": "Santali",
}


def doc_to_text(doc):
    # All four options always present in this dataset; no filtering to keep
    # alignment with doc_to_choice and doc_to_target.
    options = "\n".join(f"{CHOICES[i]}. {doc[opt]}" for i, opt in enumerate(OPTS))
    return f"Question: {doc['question_text']}\n{options}\nAnswer:"


def doc_to_choice(doc):
    # Returns all four labels — consistent with doc_to_text and doc_to_target.
    return list(CHOICES)


def doc_to_target(doc):
    # correct_answer is 'a', 'b', 'c', or 'd'
    return CHOICES.index(doc["correct_answer"].upper())


def _make_filter(subject_value):
    """Factory: returns a process_docs fn that filters by subject."""
    def process_docs(dataset):
        return dataset.filter(lambda x: x["subject"] == subject_value)
    return process_docs


# YAML-compatible named functions — generated from SUBJECT_MAP via factory.
# Adding a new language: add entry to SUBJECT_MAP and create a yaml that
# references the corresponding process_docs_<key> function below.
process_docs_bodo         = _make_filter(SUBJECT_MAP["bodo"])
process_docs_dogri        = _make_filter(SUBJECT_MAP["dogri"])
process_docs_gujarati     = _make_filter(SUBJECT_MAP["gujarati"])
process_docs_konkani      = _make_filter(SUBJECT_MAP["konkani"])
process_docs_maithili     = _make_filter(SUBJECT_MAP["maithili"])
process_docs_marathi      = _make_filter(SUBJECT_MAP["marathi"])
process_docs_nepali       = _make_filter(SUBJECT_MAP["nepali"])
process_docs_oriya        = _make_filter(SUBJECT_MAP["oriya"])
process_docs_rajasthani   = _make_filter(SUBJECT_MAP["rajasthani"])
process_docs_sanskrit     = _make_filter(SUBJECT_MAP["sanskrit"])
process_docs_sanskrit_mix = _make_filter(SUBJECT_MAP["sanskrit_mix"])
process_docs_santali      = _make_filter(SUBJECT_MAP["santali"])
