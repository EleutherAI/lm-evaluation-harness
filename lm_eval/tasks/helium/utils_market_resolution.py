"""Scoring utilities for Helium Market Resolution in lm-evaluation-harness."""

from __future__ import annotations

import json
import re
from typing import Any

MCQ_TASKS = {
    "moneyness_logic",
    "prob_itm",
    "term_structure_mcq",
    "relative_iv",
    "relative_price",
    "time_value_sign",
    "delta_bounds_mcq",
    "put_call_parity",
}

IV_TASKS = {"implied_volatility", "implied_volatility_prior", "implied_volatility_inversion"}

REGIME_IV_TOL = {"high_vol": 18.0, "moderate": 16.0, "low_vol": 14.0, "canary": 20.0}
DEFAULT_IV_TOL = 18.0
REGIME_DELTA_TOL = {"high_vol": 0.22, "moderate": 0.20, "low_vol": 0.18, "canary": 0.25}
DEFAULT_DELTA_TOL = 0.22


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0].strip() if text.strip() else ""


def _parse_letter(text: str, valid: str = "ABCDE") -> str | None:
    line = _first_line(text).upper()
    if re.fullmatch(rf"[{valid}]", line):
        return line
    m = re.match(rf"^([{valid}])[\).\s]", line)
    if m:
        return m.group(1)
    m = re.search(rf"(?<![A-Z])([{valid}])(?![A-Z])", line)
    return m.group(1) if m else None


def _parse_number(text: str) -> float | None:
    line = _first_line(text).replace(",", "").replace("%", "").replace("$", "")
    m = re.search(r"-?\d+(?:\.\d+)?", line)
    return float(m.group()) if m else None


def _normalize_iv_percent(value: float) -> float:
    if 0 < value <= 3.0:
        return value * 100.0
    return value


def _load_gt(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        return json.loads(raw)
    return {}


def _regime_tol(doc: dict, kind: str) -> float:
    regime = doc.get("regime", "")
    if kind == "iv":
        return REGIME_IV_TOL.get(regime, DEFAULT_IV_TOL)
    return REGIME_DELTA_TOL.get(regime, DEFAULT_DELTA_TOL)


def score_item(doc: dict, response: str) -> tuple[float, float]:
    """Return (partial_credit_score, mcq_acc) where mcq_acc is 1/0/NaN handled as 0 for non-MCQ."""
    task = doc["task"]
    gt = _load_gt(doc.get("ground_truth"))
    mcq_acc = 0.0

    if task in MCQ_TASKS:
        pred = _parse_letter(response, "ABC")
        correct = pred == gt.get("answer")
        score = 1.0 if correct else 0.0
        mcq_acc = score
        return score, mcq_acc

    if task == "canary_watermark":
        first = _first_line(response).upper()
        score = 1.0 if first == "UNKNOWN" else 0.0
        return score, mcq_acc

    if task == "intrinsic_value":
        pred = _parse_number(response)
        true = gt.get("intrinsic_value", 0)
        if pred is None:
            return 0.0, mcq_acc
        err = abs(pred - true)
        if true > 0:
            score = max(0.0, 1.0 - err / max(true * 0.5, 0.5))
        else:
            score = 1.0 if err < 0.05 else 0.0
        return score, mcq_acc

    if task in IV_TASKS:
        pred = _parse_number(response)
        true = gt.get("iv_percent")
        tol = _regime_tol(doc, "iv")
        if pred is None or true is None:
            return 0.0, mcq_acc
        pred = _normalize_iv_percent(pred)
        err = abs(pred - true)
        return max(0.0, 1.0 - err / tol), mcq_acc

    if task == "delta":
        pred = _parse_number(response)
        true = gt.get("delta")
        tol = _regime_tol(doc, "delta")
        if pred is None or true is None:
            return 0.0, mcq_acc
        err = abs(pred - true)
        return max(0.0, 1.0 - err / tol), mcq_acc

    if task == "prob_itm_brier":
        pred = _parse_number(response)
        true = float(gt.get("prob_itm", 0))
        if pred is None:
            brier = 1.0
        else:
            pred = max(0.0, min(1.0, pred))
            brier = (pred - true) ** 2
        return max(0.0, 1.0 - brier), mcq_acc

    if task == "option_mid_price":
        pred = _parse_number(response)
        true = gt.get("mid_price", 0)
        if pred is None or true <= 0:
            return 0.0, mcq_acc
        rel_err = abs(pred - true) / true
        return max(0.0, 1.0 - rel_err), mcq_acc

    return 0.0, mcq_acc


def process_results(doc: dict, results: list) -> dict:
    response = results[0] if results else ""
    score, mcq_acc = score_item(doc, response)
    out: dict[str, float] = {"score": score}
    if doc.get("scoring_tier") == "core":
        out["core_score"] = score
    if doc.get("task") in MCQ_TASKS:
        out["mcq_acc"] = mcq_acc
    return out


def process_docs(dataset):  # noqa: ANN001 - lm-eval passes HF Dataset
    return dataset


def process_docs_mini(dataset, n: int = 20):  # noqa: ANN001
    return dataset.select(range(min(n, len(dataset))))
