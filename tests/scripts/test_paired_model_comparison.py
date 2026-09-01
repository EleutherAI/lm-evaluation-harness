import json

import numpy as np
import pytest
from scipy.stats import binomtest

from scripts.paired_model_comparison import compare, mcnemar_exact, wilson_ci


def _write_samples(path, correctness):
    """Write an lm-eval --log_samples-style JSONL: one record per doc with doc_id + acc."""
    with open(path, "w") as fh:
        fh.writelines(json.dumps({"doc_id": i, "target": "x", "acc": float(c)}) + "\n" for i, c in enumerate(correctness))


def test_mcnemar_matches_exact_binomial():
    # deterministic discordant structure: n01=30 (A wrong, B right), n10=12 (A right, B wrong)
    a = np.array([0] * 30 + [1] * 12 + [1] * 50 + [0] * 50)
    b = np.array([1] * 30 + [0] * 12 + [1] * 50 + [0] * 50)
    r = mcnemar_exact(a, b)
    assert (r["n01"], r["n10"]) == (30, 12)
    ref = binomtest(12, 42, 0.5, alternative="two-sided").pvalue
    assert r["p_value"] == pytest.approx(ref, abs=1e-9)


def test_paired_detects_difference(tmp_path):
    a = np.array([0] * 30 + [1] * 12 + [1] * 50 + [0] * 50)
    b = np.array([1] * 30 + [0] * 12 + [1] * 50 + [0] * 50)
    _write_samples(tmp_path / "A.jsonl", a)
    _write_samples(tmp_path / "B.jsonl", b)
    r = compare(
        str(tmp_path / "A.jsonl"), str(tmp_path / "B.jsonl"), "A", "B", "acc", 0.05
    )
    assert r["n_items"] == len(a)
    assert r["mcnemar"]["p_value"] < 0.05  # paired test detects the real difference


def test_wilson_ci_stays_in_unit_interval():
    # at k=1, n=20 the naive Wald interval underflows below 0; Wilson must not
    lo, hi = wilson_ci(1, 20)
    assert 0.0 <= lo <= 1 / 20 <= hi <= 1.0


def test_small_n_warning_fires(tmp_path):
    a = np.random.default_rng(0).integers(0, 2, 50)
    b = np.random.default_rng(1).integers(0, 2, 50)
    _write_samples(tmp_path / "A.jsonl", a)
    _write_samples(tmp_path / "B.jsonl", b)
    r = compare(
        str(tmp_path / "A.jsonl"), str(tmp_path / "B.jsonl"), "A", "B", "acc", 0.05
    )
    assert r["clt_warning"] is not None
