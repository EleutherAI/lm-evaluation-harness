"""Utilities for AfriMed-QA MCQ tasks."""

from __future__ import annotations

import json
import re
from typing import Any


ANSWER_RE = re.compile(r"\b([A-E])\b", re.IGNORECASE)
INVALID_OPTION_VALUES = {"", "n/a", "na", "none", "null"}
LABELS = "ABCDE"
BASE_MCQ_NO_EXP_PROMPT = (
    # Prompt adapted from the AfriMed-QA base_mcq_no_exp prompt in
    # Olatunji et al., "AfriMed-QA: A Pan-African, Multi-Specialty,
    # Medical Question-Answering Benchmark Dataset", arXiv:2411.15640v4, 2025.
    "###Instruction: You should directly answer the question by choosing ONLY "
    "the correct option from the list of options that will be provided to you. "
    "You must only return one character between A,B,C,D,E. You MUST NOT "
    "provide any additional text or explanation."
)


def process_docs(dataset):
    """Keep evaluable expert/test MCQs and normalize options/gold labels."""
    return (
        dataset.filter(_is_mcq_test)
        .map(_process_doc)
        .filter(_is_evaluable)
    )


def doc_to_text(doc: dict[str, Any]) -> str:
    """Paper-style AfriMed-QA MCQ prompt with compacted valid options."""
    options = "\n".join(
        f"{label}. {text}"
        for label, text in zip(doc["choice_labels"], doc["choices"], strict=False)
    )
    return (
        f"{BASE_MCQ_NO_EXP_PROMPT}\n\n"
        f"###Question: {doc['query']}\n\n"
        f"###Options:\n{options}\n\n"
        "###Answer: The correct letter option is"
    )


def doc_to_target(doc: dict[str, Any]) -> str:
    """Return readable gold labels for logging; scoring is in process_results."""
    return ", ".join(doc["gold_labels"])


def process_results_gen(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    """Score a generated answer with any-correct gold labels."""
    pred = extract_mcq_answer(results[0] if results else "")
    return {"acc": float(pred in doc["gold_labels"])}


def extract_mcq_answer(output: str) -> str:
    """Extract the first standalone A-E answer from model output."""
    match = ANSWER_RE.search(output or "")
    return match.group(1).upper() if match else ""


def _is_mcq_test(doc: dict[str, Any]) -> bool:
    return doc.get("question_type") == "mcq" and doc.get("split") == "test"


def _is_evaluable(doc: dict[str, Any]) -> bool:
    return len(doc.get("choices", [])) >= 2 and len(doc.get("gold_labels", [])) >= 1


def _process_doc(doc: dict[str, Any]) -> dict[str, Any]:
    choices, option_to_label = _parse_choices(doc.get("answer_options"))
    gold_labels = _parse_gold_labels(doc.get("correct_answer", ""), option_to_label)
    query = (doc.get("question_clean") or doc.get("question") or "").strip()

    return {
        **doc,
        "query": query,
        "choices": [choice["text"] for choice in choices],
        "choice_labels": [choice["label"] for choice in choices],
        "gold_labels": gold_labels,
    }


def _parse_choices(answer_options: str | dict[str, Any] | None) -> tuple[list[dict], dict]:
    if isinstance(answer_options, str):
        options = json.loads(answer_options) if answer_options.strip() else {}
    elif isinstance(answer_options, dict):
        options = answer_options
    else:
        options = {}

    choices = []
    option_to_label = {}
    for idx in range(1, 6):
        key = f"option{idx}"
        value = _clean_option_value(options.get(key))
        if value is None:
            continue
        label = LABELS[len(choices)]
        choices.append({"source_key": key, "label": label, "text": value})
        option_to_label[key] = label
    return choices, option_to_label


def _parse_gold_labels(correct_answer: str, option_to_label: dict[str, str]) -> list[str]:
    labels = []
    for answer in (correct_answer or "").split(","):
        answer = answer.strip()
        if answer in option_to_label and option_to_label[answer] not in labels:
            labels.append(option_to_label[answer])
    return labels


def _clean_option_value(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    if value.lower() in INVALID_OPTION_VALUES:
        return None
    return value
