"""MCQ helpers for the ArabicCulturalQA lm-eval-harness tasks.

The `mcq` config of `QCRI/ArabicCulturalQA` has rows with fields
`{id, dialect, question, A, B, C, D, answer}` where `dialect` is one of:
msa, english, egyptian, gulf, levantine, maghrebi.

Each per-dialect YAML uses `process_docs: !function utils.process_docs_<dialect>`
to filter the loaded split down to its dialect.
"""


def _filter(dataset, dialect):
    return dataset.filter(lambda x: x["dialect"] == dialect)


def process_docs_msa(dataset):
    return _filter(dataset, "msa")


def process_docs_english(dataset):
    return _filter(dataset, "english")


def process_docs_egyptian(dataset):
    return _filter(dataset, "egyptian")


def process_docs_gulf(dataset):
    return _filter(dataset, "gulf")


def process_docs_levantine(dataset):
    return _filter(dataset, "levantine")


def process_docs_maghrebi(dataset):
    return _filter(dataset, "maghrebi")


_LETTER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}


def doc_to_target_mcq(doc):
    return _LETTER_TO_IDX[doc["answer"]]
