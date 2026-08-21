"""OEQ helpers for the ArabicCulturalQA lm-eval-harness tasks.

The `oeq` config of `QCRI/ArabicCulturalQA` has rows
`{id, dialect, open_question, open_answer}` where `dialect` is one of:
msa, english, egyptian, gulf, levantine, maghrebi.

`process_docs_<dialect>` filters the split to one dialect.
`process_results_oeq` is the per-doc hook that produces `(gold, pred, lang)`
triples for the BERTScore and ROUGE-L aggregators in `metrics.py`. Predictions
and gold answers are passed through the paper's normalization
(`normalize.normalize`) so the harness numbers line up with `qa_eval.py`.
"""

import importlib.util as _importlib_util
from pathlib import Path as _Path


_spec = _importlib_util.spec_from_file_location(
    "_arabicculturalqa_normalize", _Path(__file__).parent / "normalize.py"
)
_normalize = _importlib_util.module_from_spec(_spec)
_spec.loader.exec_module(_normalize)

AR_DIALECTS = {"msa", "egyptian", "gulf", "levantine", "maghrebi"}


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


def _lang_for(doc):
    d = doc.get("dialect", "msa")
    return "ar" if d in AR_DIALECTS else "en"


def process_results_oeq(doc, results):
    """Per-doc result. Normalize pred + gold, then emit `(gold, pred, lang)`
    triples that the metric aggregators consume.
    """
    raw_pred = (results[0] if results else "").strip()
    raw_gold = doc["open_answer"].strip()
    lang = _lang_for(doc)
    pred = _normalize.normalize(raw_pred, lang)
    gold = _normalize.normalize(raw_gold, lang)
    return {
        "bertscore_f1": (gold, pred, lang),
        "rougeL_f1": (gold, pred, lang),
    }
