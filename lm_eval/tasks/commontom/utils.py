from __future__ import annotations

import csv
import gzip
import re
from pathlib import Path
from typing import Any

import datasets


PROMPT = (
    "You are a cautious assistant. You carefully follow instructions. "
    "You are helpful and harmless and you follow ethical guidelines and "
    "promote positive behavior. Given a conversation, answer a yes or no "
    "question without providing any additional information.\n\n"
    "Conversation:\n{context}\n\n"
    "Question:\n{question}\n"
)


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "benchmarks" / "common-tom"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find benchmarks/common-tom from the task directory."
    )


def load_dataset(**_: Any) -> dict[str, datasets.Dataset]:
    rows: list[dict[str, Any]] = []
    for path in sorted((_repo_root() / "data" / "questions").glob("*.csv.gz")):
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return {"test": datasets.Dataset.from_list(rows)}


def doc_to_text(doc: dict[str, Any]) -> str:
    return PROMPT.format(context=doc["context"], question=doc["question"])


_YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def _normalize_answer(text: str) -> str:
    match = _YES_NO_RE.search(text or "")
    if match:
        return match.group(1).title()
    cleaned = (text or "").strip().lower()
    if cleaned in {"y", "true", "1"}:
        return "Yes"
    if cleaned in {"n", "false", "0"}:
        return "No"
    return cleaned.title()


def process_results(doc: dict[str, Any], results: list[str]) -> dict[str, Any]:
    pred = _normalize_answer(results[0] if results else "")
    gold = _normalize_answer(str(doc["answer"]))
    return {
        "acc": 1.0 if pred == gold else 0.0,
        "macro_f1": (gold, pred),
    }


def macro_f1_agg(items: list[tuple[str, str]]) -> float:
    if not items:
        return 0.0

    labels = sorted({label for item in items for label in item})
    if not labels:
        return 0.0

    def _f1_for_label(label: str) -> float:
        tp = fp = fn = 0
        for gold, pred in items:
            if pred == label and gold == label:
                tp += 1
            elif pred == label and gold != label:
                fp += 1
            elif pred != label and gold == label:
                fn += 1
        if tp == 0 and fp == 0 and fn == 0:
            return 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        if precision + recall == 0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

    return sum(_f1_for_label(label) for label in labels) / len(labels)
