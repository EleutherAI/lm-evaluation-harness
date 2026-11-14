import json
import re
import unicodedata

import datasets
import sacrebleu


def _best_chrf(pred, refs):
    pred = pred.strip().split("\n")[0]
    refs = [r for r in refs if r and r.strip()]
    return max(sacrebleu.sentence_chrf(pred, [r]).score for r in refs) if refs else 0.0


# Clue-Instruct (English)
def doc_to_text_en(doc):
    return (
        f"Text: {doc['context']}\n"
        f"Answer: {doc['keyword']}\n"
        "Write a short educational crossword clue for the answer, based only on the text.\n"
        "Clue:"
    )


def _en_refs(doc):
    try:
        return json.loads(doc["clues"])["clues"]
    except (ValueError, KeyError, TypeError):
        return [doc["clues"]]


def doc_to_target_en(doc):
    return _en_refs(doc)[0]


def process_results_en(doc, results):
    return {"chrf": _best_chrf(results[0], _en_refs(doc))}


# Italian clue generation
def doc_to_text_it(doc):
    return (
        f"Testo: {doc['Content']}\n"
        f"Risposta: {doc['answer']}\n"
        "Scrivi una breve definizione da cruciverba per la risposta, basata solo sul testo.\n"
        "Definizione:"
    )


def process_results_it(doc, results):
    return {"chrf": _best_chrf(results[0], [doc["clues"]])}


# Arabic clue generation
def doc_to_text_ar(doc):
    return (
        f"النص: {doc['Text']}\n"
        f"الجواب: {doc['keywords']}\n"
        "اكتب تعريفاً قصيراً للكلمات المتقاطعة يدل على الجواب، اعتماداً على النص فقط.\n"
        "التعريف:"
    )


def _ar_refs(doc):
    return [
        re.sub(r"^\s*CLUE\d+\s*:\s*", "", line).strip()
        for line in doc["clues"].split("\n")
        if line.strip()
    ]


def doc_to_target_ar(doc):
    return _ar_refs(doc)[0]


def process_results_ar(doc, results):
    return {"chrf": _best_chrf(results[0], _ar_refs(doc))}


# Italian clue solving (Crossword-Surprisal-IT)
def process_docs_solving(dataset):
    # The source table has one row per clue, model and surprisal measure; keep one row per clue.
    df = (
        dataset.to_pandas()
        .drop_duplicates("item")
        .sort_values("item", key=lambda s: s.astype(int))
    )
    return datasets.Dataset.from_pandas(df, preserve_index=False)


def doc_to_text_solving(doc):
    return (
        f"Definizione di cruciverba: {doc['clue']}\n"
        f"Lunghezza della risposta: {int(float(doc['answer_length']))} lettere\n"
        "Risposta:"
    )


def _norm(text):
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z]", "", text)


def process_results_solving(doc, results):
    pred = results[0].strip().split("\n")[0].split()
    pred = pred[0] if pred else ""
    return {"exact_match": float(_norm(pred) == _norm(doc["solution"]))}
