"""Utility functions for jfinqa lm-evaluation-harness integration.

Referenced by YAML task configs via ``!function utils.<name>``.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any


def doc_to_text(doc: dict[str, Any]) -> str:
    """Format a dataset row as a prompt for the LLM."""
    parts: list[str] = []

    # Pre-text paragraphs
    pre_text = doc.get("pre_text", [])
    if pre_text:
        parts.append("\n".join(pre_text))

    # Table as markdown
    headers = doc.get("table_headers", [])
    rows = doc.get("table_rows", [])

    if headers:
        header_line = "| " + " | ".join(str(h) for h in headers) + " |"
        sep_line = "| " + " | ".join("---" for _ in headers) + " |"
        row_lines = ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
        parts.append("\n".join([header_line, sep_line, *row_lines]))

    # Post-text paragraphs
    post_text = doc.get("post_text", [])
    if post_text:
        parts.append("\n".join(post_text))

    # Question
    question = doc.get("question", "")
    parts.append(f"Question: {question}\nAnswer:")

    return "\n\n".join(parts)


def process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    """Score a model response against the gold answer."""
    gold = doc.get("answer", "")
    predicted = results[0] if results else ""
    predicted = _extract_answer(predicted)

    em = 1.0 if _normalize(predicted) == _normalize(gold) else 0.0
    nm = 1.0 if _numerical_match(predicted, gold) else 0.0

    return {"exact_match": em, "numerical_match": nm}


def _extract_answer(text: str) -> str:
    """Extract the answer from model output."""
    match = re.search(r"(?:Answer|answer|A)\s*[:\uff1a]\s*(.+)", text)
    if match:
        return match.group(1).strip()
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _normalize(text: str) -> str:
    """Normalize an answer for comparison."""
    s = text.strip()
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"^[△▲]", "-", s)
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    if s.endswith("しました"):
        s = s.removesuffix("しました")
    elif s.endswith("した"):
        s = s.removesuffix("した")
    return s.lower().strip()


_UNIT_SUFFIXES = (
    "百万円",
    "千円",
    "億円",
    "兆円",
    "円",
    "ドル",
    "ポイント",
    "pt",
    "bps",
)

_KANJI_MULTIPLIERS: dict[str, int] = {
    "千": 1_000,
    "百万": 1_000_000,
    "億": 100_000_000,
    "兆": 1_000_000_000_000,
}


def _try_parse_number(text: str) -> float | None:
    """Try to extract a number from text."""
    s = _normalize(text)
    for suffix in _UNIT_SUFFIXES:
        s = s.removesuffix(suffix)

    for kanji, multiplier in _KANJI_MULTIPLIERS.items():
        if kanji in s:
            num_part = s.replace(kanji, "").strip()
            num_part = re.sub(r"[^\d.\-+]", "", num_part)
            try:
                return float(num_part) * multiplier
            except ValueError:
                return None

    is_percent = s.endswith("%")
    if is_percent:
        s = s.removesuffix("%")

    s = re.sub(r"[^\d.\-+]", "", s)
    try:
        return float(s)
    except ValueError:
        return None


def _numerical_match(predicted: str, gold: str, tolerance: float = 0.01) -> bool:
    """Check numerical equivalence with 1% tolerance."""
    pred_num = _try_parse_number(predicted)
    gold_num = _try_parse_number(gold)

    if pred_num is None or gold_num is None:
        return _normalize(predicted) == _normalize(gold)

    if gold_num == 0:
        return pred_num == 0

    return abs(pred_num - gold_num) / abs(gold_num) <= tolerance
