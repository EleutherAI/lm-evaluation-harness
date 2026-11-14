import re
import unicodedata

import datasets


def _subset(*names):
    def _process(dataset: datasets.Dataset) -> datasets.Dataset:
        return dataset.filter(lambda d: d["task"] in names)

    return _process


process_multiple_choice = _subset("multiple_choice")
process_true_false = _subset("true_false_1", "true_false_2")
process_cloze = _subset("cloze")
process_explanation = _subset("explanation")
process_counterfactual = _subset("counterfactual")
process_cause_effect = _subset("cause_effect")


def _context(doc):
    return f"متن: {doc['context']}\n"


def doc_to_text_multiple_choice(doc):
    return f"{_context(doc)}پرسش: {doc['question']}\nپاسخ:"


def doc_to_text_true_false(doc):
    return f"{_context(doc)}گزاره: {doc['question']}\nآیا این گزاره بر اساس متن درست است؟\nپاسخ:"


def doc_to_text_cloze(doc):
    return f"{_context(doc)}جمله: {doc['question']}\nعبارتی که به جای «_____» می‌آید:"


def doc_to_text_explanation(doc):
    return f"{_context(doc)}پرسش: {doc['question']}\nپاسخ کوتاه (حداکثر دو جمله):"


def doc_to_text_counterfactual(doc):
    return f"{_context(doc)}پرسش خلاف واقع: {doc['question']}\nپاسخ کوتاه (حداکثر دو جمله):"


def doc_to_text_cause_effect(doc):
    return f"{_context(doc)}علت: {doc['question']}\nمعلول (حداکثر دو جمله):"


def _norm(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("‌", " ").replace("ي", "ی").replace("ك", "ک")
    text = re.sub(r"[«»\"'.,،؛:!?؟()\[\]]", " ", text)
    return " ".join(text.split())


def process_results_cloze(doc, results):
    pred = results[0].strip().split("\n")[0]
    return {"exact_match": float(_norm(pred) == _norm(doc["answer"]))}
