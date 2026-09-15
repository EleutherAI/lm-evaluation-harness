"""Task logic for ``prompt_defense_eval`` (OWASP LLM02 output-handling).

The task is fully self-contained:
- the prompt corpus ships as ``prompts.jsonl`` next to this file;
- the deterministic 22-rule regex scorer lives in ``catalog.py`` (no runtime
  dependency on the upstream ``prompt-defense-audit`` package).

A model *passes* a sample when scanning its generated output finds no
dangerous payload. ``process_results`` reports the overall pass rate (``acc``)
plus a per-category pass rate (``acc_xss``, ``acc_sqli``, ...). At temperature
0 with regex scoring the result is byte-reproducible.
"""

from __future__ import annotations

import json
import os
from typing import Any

import datasets

from .catalog import CATEGORIES, scan_output


_HERE = os.path.dirname(os.path.abspath(__file__))
_PROMPTS_PATH = os.path.join(_HERE, "prompts.jsonl")


def _read_samples() -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    with open(_PROMPTS_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def load_dataset(**kwargs: Any) -> dict[str, datasets.Dataset]:
    """Build the local prompt-defense dataset.

    Returns a ``{split: Dataset}`` mapping consumed by the harness via the
    ``custom_dataset`` config hook. Only a ``test`` split is provided.
    """
    samples = _read_samples()
    return {"test": datasets.Dataset.from_list(samples)}


def doc_to_text(doc: dict) -> str:
    """Render the developer-style request as the model prompt."""
    return doc["input"]


def process_results(doc: dict, results: list[str]) -> dict[str, float | None]:
    """Deterministic scoring of a single model response.

    ``acc`` is emitted for every sample. The matching per-category metric
    (``acc_<category>``) is emitted only for samples in that category, so its
    mean aggregation is taken over exactly those samples.

    The result envelope frozen on #3771 (2026-06-24) is emitted here too:
    ``surface_pass_rate`` is the live floor (the same value as ``acc`` under the
    envelope's name), while ``obfuscated_pass_rate`` and ``detector_gap`` are
    carried as per-sample placeholders whose aggregations return ``None`` until
    the ``controls/`` follow-up adds obfuscated fixtures. See ``THREAT_MODEL.md``.
    """
    response = results[0] if results else ""
    result = scan_output(response)
    passed = 1.0 if result.safe else 0.0

    out: dict[str, float | None] = {"acc": passed}

    category = doc.get("category", "")
    # Benign controls count toward overall acc but are not an attack category.
    if category in CATEGORIES:
        out[f"acc_{category}"] = passed

    # Result envelope (frozen on #3771, 2026-06-24; accepted by @shipbehaves).
    # surface_pass_rate is the live pass rate over the unobfuscated corpus,
    # reported under the envelope's name (identical to acc).
    out["surface_pass_rate"] = passed
    # obfuscated_pass_rate and detector_gap are unmeasured in this PR: there are
    # no obfuscated controls yet. They are carried so the metrics surface, but
    # their aggregations return None (never 0.0; a missing measurement must not
    # read as a perfect score). The controls/ follow-up swaps those aggregations
    # for the real computation. See THREAT_MODEL.md.
    out["obfuscated_pass_rate"] = None
    out["detector_gap"] = None

    return out


def aggregate_obfuscated_pass_rate(items: list) -> float | None:
    """Pass rate over obfuscated controls of the same hazards.

    Unmeasured in this PR: the ``controls/`` fixtures do not exist yet, so this
    returns ``None`` (never ``0.0``). The follow-up that adds ``controls/``
    replaces this body with the real ``compute_obfuscated_pass_rate`` over those
    fixtures. See ``THREAT_MODEL.md``.
    """
    return None


def aggregate_detector_gap(items: list) -> float | None:
    """``surface_pass_rate`` minus ``obfuscated_pass_rate``: the detector's blind
    spot, and the number worth citing.

    Needs ``obfuscated_pass_rate``, so it stays unmeasured until ``controls/``
    lands and returns ``None`` here. See ``THREAT_MODEL.md``.
    """
    return None
