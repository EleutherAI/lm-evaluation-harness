"""Utility functions for ZeroSCROLLS tasks.

ZeroSCROLLS: A Zero-Shot Benchmark for Long Text Understanding
https://arxiv.org/abs/2305.14196
"""

import math
import re
import string
from typing import List, Optional


# ---------------------------------------------------------------------------
# Dataset preprocessing
# ---------------------------------------------------------------------------


def process_docs(docs):
    """Add 'question' and 'context' fields extracted from 'input' via index fields.

    ZeroSCROLLS stores the full formatted input in a single 'input' field.
    For QA/summarization tasks the query (if any) is prepended to the document
    with '\n\n' as separator.  The byte-offset fields identify each part.
    """

    def _process(doc):
        text = doc["input"]
        n = len(text)

        # Use explicit None checks: 0 is a valid start index but is falsy.
        raw_qs = doc.get("query_start_index")
        raw_qe = doc.get("query_end_index")
        raw_ds = doc.get("document_start_index")
        raw_de = doc.get("document_end_index")

        qs = -1 if raw_qs is None else int(raw_qs)
        qe = -1 if raw_qe is None else int(raw_qe)
        ds = 0 if raw_ds is None else int(raw_ds)
        de = n if raw_de is None else int(raw_de)

        question = text[qs:qe] if 0 <= qs < qe <= n else ""
        context = text[ds : min(de, n)]

        return {**doc, "question": question, "context": context}

    return docs.map(_process)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def _normalize_answer(text: str) -> str:
    """Lower-case, strip punctuation, articles and extra whitespace."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = " ".join(text.split())
    return text


def _get_tokens(text: str) -> List[str]:
    return _normalize_answer(text).split()


def _token_f1(prediction: str, reference: str) -> float:
    pred_toks = _get_tokens(prediction)
    ref_toks = _get_tokens(reference)
    if not pred_toks or not ref_toks:
        return 1.0 if pred_toks == ref_toks else 0.0
    common = set(pred_toks) & set(ref_toks)
    num_common = sum(min(pred_toks.count(t), ref_toks.count(t)) for t in common)
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_toks)
    recall = num_common / len(ref_toks)
    return 2 * precision * recall / (precision + recall)


def _refs(doc: dict) -> List[str]:
    """Return a list of reference strings from a doc's 'output' field."""
    out = doc.get("output")
    if out is None:
        return []
    if isinstance(out, list):
        return [str(r) for r in out if r is not None]
    return [str(out)]


# ---------------------------------------------------------------------------
# ROUGE – geometric mean of ROUGE-1, ROUGE-2, ROUGE-L (ZeroSCROLLS metric)
# ---------------------------------------------------------------------------


def process_rouge(doc: dict, results: list) -> dict:
    """Geometric mean of ROUGE-1/2/L F-measures, scaled to 0-100."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        raise ImportError(
            "rouge-score is required for ZeroSCROLLS tasks. "
            "Install with: pip install rouge-score"
        )

    prediction = results[0].strip()
    refs = _refs(doc)
    if not prediction or not refs:
        return {"rouge_geomean": 0.0}

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    best = 0.0
    for ref in refs:
        s = scorer.score(ref, prediction)
        r1 = s["rouge1"].fmeasure
        r2 = s["rouge2"].fmeasure
        rL = s["rougeL"].fmeasure
        product = r1 * r2 * rL
        geomean = product ** (1.0 / 3.0) if product > 0 else 0.0
        best = max(best, geomean)

    return {"rouge_geomean": best * 100}


# ---------------------------------------------------------------------------
# Token-level F1 (QA tasks)
# ---------------------------------------------------------------------------


def process_qa_f1(doc: dict, results: list) -> dict:
    """Token-level F1, max over multiple references, scaled to 0-100."""
    prediction = results[0].strip()
    refs = _refs(doc)
    if not refs:
        return {"f1": 0.0}
    score = max(_token_f1(prediction, ref) for ref in refs)
    return {"f1": score * 100}


# ---------------------------------------------------------------------------
# Accuracy – QuALITY (multiple-choice, letter A/B/C/D)
# ---------------------------------------------------------------------------


def process_accuracy(doc: dict, results: list) -> dict:
    """Extract A/B/C/D from model output and compare to reference letter."""
    prediction = results[0].strip()
    refs = _refs(doc)
    if not refs:
        return {"acc": 0.0}
    ref = refs[0]

    pred_match = re.search(r"\b([ABCD])\b", prediction.upper())
    ref_match = re.search(r"\b([ABCD])\b", ref.upper())

    pred_letter = pred_match.group(1) if pred_match else ""
    ref_letter = ref_match.group(1) if ref_match else ref.strip()[:1].upper()

    return {"acc": 1.0 if pred_letter == ref_letter else 0.0}


# ---------------------------------------------------------------------------
# Exponential Similarity – SpaceDigest
# ---------------------------------------------------------------------------


def _extract_fraction(text: str) -> Optional[float]:
    """Parse the first number from *text* and normalise to [0, 1]."""
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    if not nums:
        return None
    val = float(nums[0])
    # Accept both 0-1 and 0-100 representations
    if val > 1.0:
        val /= 100.0
    return max(0.0, min(1.0, val))


def process_space_digest(doc: dict, results: list) -> dict:
    """ES(p, p_hat) = 2^(-10 * |p - p_hat|), scaled to 0-100.

    p is the true fraction of reviews that recommend the hotel;
    p_hat is the model's predicted fraction.  A perfect prediction
    scores 100; a 10-percentage-point error scores ~50.
    """
    prediction = results[0].strip()
    refs = _refs(doc)
    if not refs:
        return {"es": 0.0}

    p = _extract_fraction(refs[0])
    p_hat = _extract_fraction(prediction)
    if p is None or p_hat is None:
        return {"es": 0.0}

    return {"es": 2 ** (-10 * abs(p - p_hat)) * 100}


# ---------------------------------------------------------------------------
# Concordance Index – BookSumSort
# ---------------------------------------------------------------------------


def process_book_sum_sort(doc: dict, results: list) -> dict:
    """Fraction of chapter-pairs correctly ordered (concordance index).

    The model outputs a sequence of chapter numbers in the order it thinks
    they appear in the book.  We compare against the reference ordering.
    Score is (correctly ordered pairs) / (total pairs), scaled to 0-100.
    A random permutation expected to score ~50.
    """
    prediction = results[0].strip()
    refs = _refs(doc)
    if not refs:
        return {"concordance": 0.0}

    def _extract_order(text: str) -> List[int]:
        return [int(x) for x in re.findall(r"\d+", text)]

    pred_order = _extract_order(prediction)
    ref_order = _extract_order(refs[0])

    if len(pred_order) < 2 or not ref_order:
        return {"concordance": 0.0}

    ref_rank = {val: i for i, val in enumerate(ref_order)}
    total, correct = 0, 0
    for i in range(len(pred_order)):
        for j in range(i + 1, len(pred_order)):
            a, b = pred_order[i], pred_order[j]
            if a not in ref_rank or b not in ref_rank:
                continue
            total += 1
            if ref_rank[a] < ref_rank[b]:
                correct += 1

    if total == 0:
        return {"concordance": 0.0}
    return {"concordance": (correct / total) * 100}
