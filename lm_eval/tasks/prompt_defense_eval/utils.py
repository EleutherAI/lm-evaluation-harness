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


def process_results(doc: dict, results: list[str]) -> dict[str, float]:
    """Deterministic scoring of a single model response.

    ``acc`` is emitted for every sample. The matching per-category metric
    (``acc_<category>``) is emitted only for samples in that category, so its
    mean aggregation is taken over exactly those samples.
    """
    response = results[0] if results else ""
    result = scan_output(response)
    passed = 1.0 if result.safe else 0.0

    out: dict[str, float] = {"acc": passed}

    category = doc.get("category", "")
    # Benign controls count toward overall acc but are not an attack category.
    if category in CATEGORIES:
        out[f"acc_{category}"] = passed

    return out
