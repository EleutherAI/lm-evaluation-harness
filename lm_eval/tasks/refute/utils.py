"""Helpers for the REFUTE lm-evaluation-harness tasks."""

from __future__ import annotations

from typing import Any


def doc_to_target(doc: dict[str, Any]) -> int:
    """Gold choice index for refute_soundness: 0 = sound, 1 = flawed."""
    return 0 if str(doc.get("label", "")).strip().lower() == "sound" else 1
