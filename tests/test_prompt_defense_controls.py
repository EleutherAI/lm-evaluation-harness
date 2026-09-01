"""Correctness tests for the ``prompt_defense_eval`` blind-spot control set.

The control set (``controls/controls.jsonl``) pairs each hazard's canonical form
(which the catalog must fire on) with obfuscated equivalents (which it must miss).
These tests pin, against the real ``catalog.scan_output``:

1. every ``canonical_positive`` is detected (fires its ``targets_rule``);
2. every ``obfuscated`` variant evades (scans safe) *after a json.loads round-trip*
   — so an escape-encoded fixture that silently decoded into a firing form fails
   here rather than shipping as a false evasion;
3. the ``sentinel_id`` invariant: present on every control row and absent from
   every live prompt in ``prompts.jsonl``, so the control/live gate holds;
4. the measured detector_gap (1.0 overall and per family), the number the
   ``controls/`` follow-up exists to produce.
"""

from __future__ import annotations

import json
import os

import pytest

from lm_eval.tasks.prompt_defense_eval.catalog import scan_output
from lm_eval.tasks.prompt_defense_eval.utils import _read_samples

_TASK_DIR = os.path.dirname(
    os.path.abspath(
        __import__("lm_eval.tasks.prompt_defense_eval.utils", fromlist=["utils"]).__file__
    )
)
_CONTROLS_PATH = os.path.join(_TASK_DIR, "controls", "controls.jsonl")

_CATALOG_RULE_IDS = {
    "xss-script-tag", "xss-event-handler", "xss-javascript-uri", "xss-data-uri-html",
    "xss-iframe-srcdoc", "xss-svg-script", "sqli-destructive", "sqli-union",
    "sqli-comment-bypass", "shell-pipe-exec", "shell-destructive", "shell-reverse",
    "shell-env-exfil", "path-traversal", "credential-api-key", "credential-private-key",
    "credential-connection-string", "credential-jwt", "markdown-link-injection",
    "markdown-image-tracking", "code-eval", "code-python-import",
}


def _read_controls() -> list[dict]:
    rows = []
    with open(_CONTROLS_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))  # round-trip is the point
    return rows


CONTROLS = _read_controls()


def _fired(text: str) -> bool:
    return not scan_output(text).safe


def _rule_ids(text: str) -> set[str]:
    return {t.id for t in scan_output(text).threats}


class TestControlSetShape:
    def test_nonempty(self):
        assert len(CONTROLS) >= 4

    def test_required_fields(self):
        for r in CONTROLS:
            assert {"id", "sentinel_id", "category", "targets_rule",
                    "canonical_positive", "obfuscated"} <= set(r)
            assert isinstance(r["obfuscated"], list) and r["obfuscated"]

    def test_targets_rule_is_a_real_catalog_rule(self):
        for r in CONTROLS:
            assert r["targets_rule"] in _CATALOG_RULE_IDS

    def test_row_ids_and_sentinel_ids_unique(self):
        ids = [r["id"] for r in CONTROLS]
        sents = [r["sentinel_id"] for r in CONTROLS]
        assert len(ids) == len(set(ids))
        assert len(sents) == len(set(sents))


class TestCanonicalPositivesFire:
    def test_each_canonical_is_detected_by_its_target_rule(self):
        for r in CONTROLS:
            text = r["canonical_positive"]
            assert _fired(text), f"{r['id']}: canonical must fire"
            assert r["targets_rule"] in _rule_ids(text), (
                f"{r['id']}: expected {r['targets_rule']} to fire on the canonical form"
            )


class TestObfuscatedVariantsEvade:
    def test_each_obfuscated_variant_scans_safe(self):
        for r in CONTROLS:
            for i, obf in enumerate(r["obfuscated"]):
                assert not _fired(obf), (
                    f"{r['id']} obfuscated[{i}] must evade the catalog but fired "
                    f"({_rule_ids(obf)}); check for accidental escape decoding"
                )


class TestSentinelInvariant:
    def test_every_control_row_has_a_sentinel_id(self):
        assert all(r.get("sentinel_id") for r in CONTROLS)

    def test_no_live_prompt_carries_a_sentinel_id(self):
        for s in _read_samples():
            assert "sentinel_id" not in s


class TestMeasuredGap:
    def test_overall_detector_gap_is_one(self):
        canon = [r["canonical_positive"] for r in CONTROLS]
        obf = [o for r in CONTROLS for o in r["obfuscated"]]
        surface = sum(_fired(c) for c in canon) / len(canon)
        obfusc = sum(_fired(o) for o in obf) / len(obf)
        assert surface == 1.0
        assert obfusc == 0.0
        assert surface - obfusc == 1.0

    @pytest.mark.parametrize("row", CONTROLS, ids=[r["id"] for r in CONTROLS])
    def test_per_family_gap_is_one(self, row):
        canon_fires = 1.0 if _fired(row["canonical_positive"]) else 0.0
        obf_fires = sum(_fired(o) for o in row["obfuscated"]) / len(row["obfuscated"])
        assert canon_fires - obf_fires == 1.0
