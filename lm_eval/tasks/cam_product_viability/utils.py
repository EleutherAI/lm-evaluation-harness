"""Utilities for the cam_product_viability tasks.

Scoring follows the CAM Pathways benchmark's reference scorer (score.py):
verdict accuracy, margin accuracy (profit figure within 1 JPY), and
report-form usage — here, exact match of the canonical report line.

Canonical output forms (stated verbatim in every prompt):
  complete case:  CAM verdict: <TEST_CANDIDATE|HOLD|REJECT> — profit ¥<profit> per order
  partial case:   CAM verdict: HOLD — missing: <item names, comma-separated>
"""
import re
from typing import Dict, List

COMMITTED_RE = re.compile(r"^CAM verdict:\s*(TEST_CANDIDATE|HOLD|REJECT)\b")
PROFIT_RE = re.compile(r"profit (-?)¥([\d,]+(?:\.\d+)?)")

MONEY_KEYS = ["price", "cost", "shipping", "fee", "returnReserve", "otherVariable"]

RULES = """Decide the product viability verdict under the CAM standard.

The six money items are price, cost (item cost), shipping, fee (marketplace/payment fees), returnReserve (expected return cost) and otherVariable (other variable costs). Count every item, including the return reserve.
profit = price - (cost + shipping + fee + returnReserve + otherVariable)
- If profit is 0 or less, the verdict is REJECT.
- Otherwise, if profit is below targetProfit, or deliveryDays is above maxDeliveryDays, or estimateAgeDays is above maxEstimateAgeDays, the verdict is HOLD.
- Otherwise the verdict is TEST_CANDIDATE.
- If any money item is not provided, never guess it: the verdict is HOLD and the missing items must be named.

Answer with exactly one line, in one of these two forms:
CAM verdict: <TEST_CANDIDATE|HOLD|REJECT> \u2014 profit \u00a5<profit> per order
CAM verdict: HOLD \u2014 missing: <missing item names, comma-separated>"""


def _fmt(x):
    if x is None:
        return "not provided"
    if isinstance(x, float) and x == int(x):
        return str(int(x))
    return str(x)


def doc_to_text(doc: dict) -> str:
    """Render the canonical prompt from the doc's raw case fields. This is the
    single source of truth for the prompt text (prepare_dataset.py emits only
    the compact fields, so the dataset file stays small and the prompt cannot
    drift between rows)."""
    lines = [RULES, "", "Case (all amounts in JPY per order):"]
    for key in MONEY_KEYS:
        lines.append(f"{key}: {_fmt(doc[key])}")
    if doc["subset"] != "partial":
        lines.append(f"targetProfit: {_fmt(doc['targetProfit'])}")
        lines.append(f"deliveryDays: {_fmt(doc['deliveryDays'])}")
        lines.append(f"maxDeliveryDays: {_fmt(doc['maxDeliveryDays'])}")
        lines.append(f"estimateAgeDays: {_fmt(doc['estimateAgeDays'])}")
        lines.append(f"maxEstimateAgeDays: {_fmt(doc['maxEstimateAgeDays'])}")
    lines.append("")
    lines.append("Verdict:")
    return "\n".join(lines)


def process_docs_complete(dataset):
    return dataset.filter(lambda doc: doc["subset"] == "complete")


def process_docs_partial(dataset):
    return dataset.filter(lambda doc: doc["subset"] == "partial")


def _candidate_line(text: str) -> str:
    """The line the model committed to: the first 'CAM verdict:' line whose
    verdict slot holds an actual verdict token and which is not a template
    echo. (The prompt shows the two answer forms with <placeholders>; a model
    that merely echoes those forms has committed to nothing. The complete-case
    form puts '<' in the verdict slot, and both forms carry '<...>'
    placeholders in the body, so a committed line must match the strict
    pattern AND contain no angle brackets.) Falls back to the first
    'CAM verdict:' line, then to the first non-empty line."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    for ln in lines:
        if COMMITTED_RE.match(ln) and "<" not in ln and ">" not in ln:
            return ln
    for ln in lines:
        if ln.startswith("CAM verdict:"):
            return ln
    return lines[0] if lines else ""


def _verdict_accuracy(doc: dict, text: str) -> int:
    m = COMMITTED_RE.match(_candidate_line(text))
    return 1 if m and m.group(1) == doc["expected_verdict"] else 0


def _line_exact_match(doc: dict, text: str) -> int:
    return 1 if _candidate_line(text) == doc["target"] else 0


def process_results_complete(doc: dict, results: List[str]) -> Dict[str, float]:
    text = results[0]
    line = _candidate_line(text)
    margin_acc = 0
    m = PROFIT_RE.search(line)
    if m and doc["expected_margin"] is not None:
        parsed = float(m.group(2).replace(",", "")) * (-1 if m.group(1) else 1)
        if abs(parsed - doc["expected_margin"]) < 1:  # score.py tolerance: within 1 JPY
            margin_acc = 1
    return {
        "verdict_accuracy": _verdict_accuracy(doc, text),
        "line_exact_match": _line_exact_match(doc, text),
        "margin_accuracy": margin_acc,
    }


def process_results_partial(doc: dict, results: List[str]) -> Dict[str, float]:
    text = results[0]
    missing = [s for s in doc["missing_items"].split(", ") if s]
    named = sum(
        1 for name in missing if re.search(rf"\b{re.escape(name)}\b", text)
    )
    return {
        "verdict_accuracy": _verdict_accuracy(doc, text),
        "line_exact_match": _line_exact_match(doc, text),
        "missing_named_rate": named / len(missing) if missing else 1.0,
    }
