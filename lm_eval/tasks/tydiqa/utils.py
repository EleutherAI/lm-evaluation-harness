"""
Evaluation utilities for TyDiQA Gold Passage task.

Based on the official MLQA evaluation script pattern used in this harness,
extended for the 9 TyDiQA Gold Passage languages. Normalization follows the
approach from:
- MLQA: https://github.com/facebookresearch/MLQA
- SQuAD: https://rajpurkar.github.io/SQuAD-explorer/
"""

import re
import string
import sys
import unicodedata
from collections import Counter

import datasets


PUNCT = {
    chr(i)
    for i in range(sys.maxunicode)
    if unicodedata.category(chr(i)).startswith("P")
}.union(string.punctuation)


def whitespace_tokenize(text):
    return text.split()


def normalize_answer(s, lang):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text, lang):
        if lang == "en":
            return re.sub(r"\b(a|an|the)\b", " ", text)
        elif lang == "ar":
            return re.sub(r"(?:^|\s)ال", " ", text)
        else:
            return text

    def white_space_fix(text):
        tokens = whitespace_tokenize(text)
        return " ".join([t for t in tokens if t.strip() != ""])

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in PUNCT)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s)), lang))


def f1_score(prediction, ground_truth, lang):
    prediction_tokens = normalize_answer(prediction, lang).split()
    ground_truth_tokens = normalize_answer(ground_truth, lang).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth, lang):
    return normalize_answer(prediction, lang) == normalize_answer(ground_truth, lang)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, lang):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth, lang)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def process_docs_lang(dataset: datasets.Dataset, lang: str) -> datasets.Dataset:
    """Filter dataset by language and restructure answer fields."""

    def _is_lang(doc):
        return doc["id"].startswith(f"{lang}-")

    def _process_doc(doc):
        return {
            "id": doc["id"],
            "title": doc["title"],
            "context": doc["context"],
            "question": doc["question"],
            "answers": doc["answers"]["text"],
        }

    return dataset.filter(_is_lang).map(_process_doc)


def process_results_lang(doc, results, lang):
    ground_truths = doc["answers"]
    prediction = results[0].strip()
    exact_match = metric_max_over_ground_truths(
        exact_match_score, prediction, ground_truths, lang
    )
    f1 = metric_max_over_ground_truths(f1_score, prediction, ground_truths, lang)
    return {"exact_match": exact_match, "f1": f1}


# Language-specific process_docs functions
def process_docs_ar(dataset):
    return process_docs_lang(dataset, "arabic")


def process_docs_bn(dataset):
    return process_docs_lang(dataset, "bengali")


def process_docs_en(dataset):
    return process_docs_lang(dataset, "english")


def process_docs_fi(dataset):
    return process_docs_lang(dataset, "finnish")


def process_docs_id(dataset):
    return process_docs_lang(dataset, "indonesian")


def process_docs_ko(dataset):
    return process_docs_lang(dataset, "korean")


def process_docs_ru(dataset):
    return process_docs_lang(dataset, "russian")


def process_docs_sw(dataset):
    return process_docs_lang(dataset, "swahili")


def process_docs_te(dataset):
    return process_docs_lang(dataset, "telugu")


# Language-specific process_results functions
def process_results_ar(doc, results):
    return process_results_lang(doc, results, "ar")


def process_results_bn(doc, results):
    return process_results_lang(doc, results, "bn")


def process_results_en(doc, results):
    return process_results_lang(doc, results, "en")


def process_results_fi(doc, results):
    return process_results_lang(doc, results, "fi")


def process_results_id(doc, results):
    return process_results_lang(doc, results, "id")


def process_results_ko(doc, results):
    return process_results_lang(doc, results, "ko")


def process_results_ru(doc, results):
    return process_results_lang(doc, results, "ru")


def process_results_sw(doc, results):
    return process_results_lang(doc, results, "sw")


def process_results_te(doc, results):
    return process_results_lang(doc, results, "te")
