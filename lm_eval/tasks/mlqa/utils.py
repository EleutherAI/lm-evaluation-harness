"""
Code based on Official evaluation script for the MLQA dataset.
Repo: https://github.com/facebookresearch/MLQA/blob/main/mlqa_evaluation_v1.py
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
WHITESPACE_LANGS = ["en", "es", "hi", "vi", "de", "ar"]
MIXED_SEGMENTATION_LANGS = ["zh"]


def whitespace_tokenize(text):
    return text.split()


def mixed_segmentation(text):
    segs_out = []
    temp_str = ""
    for char in text:
        if re.search(r"[\u4e00-\u9fa5]", char) or char in PUNCT:
            if temp_str != "":
                ss = whitespace_tokenize(temp_str)
                segs_out.extend(ss)
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char

    if temp_str != "":
        ss = whitespace_tokenize(temp_str)
        segs_out.extend(ss)

    return segs_out


def normalize_answer(s, lang):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text, lang):
        if lang == "en":
            return re.sub(r"\b(a|an|the)\b", " ", text)
        elif lang == "es":
            return re.sub(r"\b(un|una|unos|unas|el|la|los|las)\b", " ", text)
        elif lang == "hi":
            return text  # Hindi does not have formal articles
        elif lang == "vi":
            return re.sub(r"\b(của|là|cái|chiếc|những)\b", " ", text)
        elif lang == "de":
            return re.sub(
                r"\b(ein|eine|einen|einem|eines|einer|der|die|das|den|dem|des)\b",
                " ",
                text,
            )
        elif lang == "ar":
            return re.sub(r"\sال^|ال", " ", text)
        elif lang == "zh":
            return text  # Chinese does not have formal articles
        else:
            raise Exception("Unknown Language {}".format(lang))

    def white_space_fix(text, lang):
        if lang in WHITESPACE_LANGS:
            tokens = whitespace_tokenize(text)
        elif lang in MIXED_SEGMENTATION_LANGS:
            tokens = mixed_segmentation(text)
        else:
            raise Exception("Unknown Language {}".format(lang))
        return " ".join([t for t in tokens if t.strip() != ""])

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in PUNCT)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s)), lang), lang)


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


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        out_doc = {
            "context": doc["context"],
            "question": doc["question"],
            "answers": doc["answers"]["text"],
        }
        return out_doc

    return dataset.map(_process_doc)


# Base function
def process_results_lang(doc, results, lang):
    ground_truths = doc["answers"]
    prediction = results[0].strip()
    exact_match = metric_max_over_ground_truths(
        exact_match_score, prediction, ground_truths, lang
    )
    f1 = metric_max_over_ground_truths(f1_score, prediction, ground_truths, lang)
    return {"exact_match": exact_match, "f1": f1}


# Language Wrapper functions
def process_results_en(doc, results):
    return process_results_lang(doc, results, "en")


def process_results_es(doc, results):
    return process_results_lang(doc, results, "es")


def process_results_hi(doc, results):
    return process_results_lang(doc, results, "hi")


def process_results_vi(doc, results):
    return process_results_lang(doc, results, "vi")


def process_results_de(doc, results):
    return process_results_lang(doc, results, "de")


def process_results_ar(doc, results):
    return process_results_lang(doc, results, "ar")


def process_results_zh(doc, results):
    return process_results_lang(doc, results, "zh")
