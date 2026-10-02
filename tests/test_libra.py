"""LIBRA scores follow the benchmark reference evaluator."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


pytest.importorskip("pymorphy3")

from lm_eval.tasks.libra import utils


@pytest.mark.parametrize(
    ("prediction", "reference", "expected"),
    [
        ("мир", "рим", 0.0),
        ("cat", "act", 0.0),
        ("красный кот", "красный пёс", 0.5),
        ("кот кот", "кот", 2 / 3),
        ("кот собака", "собака кот", 1.0),
        ("", "кот", 0.0),
        ("кот", "", 0.0),
    ],
)
def test_libra_f1_counts_tokens(prediction, reference, expected):
    assert utils.f1_score(prediction, reference) == pytest.approx(expected)


def test_libra_aggregate_f1_normalizes_before_token_scoring():
    results = [
        {"pred_answer": "КОТЫ!", "answers": ["кот"], "length": "8p"},
        {"pred_answer": "мир", "answers": ["рим"], "length": "8p"},
        {"pred_answer": "красный кот", "answers": ["красный пёс"], "length": "16p"},
    ]
    assert utils.aggregate_results_f1(results) == {"8p": 0.5, "16p": 0.5}


@pytest.mark.parametrize("hash_seed", ["0", "1", "2"])
def test_libra_best_reference_does_not_depend_on_hash_seed(hash_seed):
    # Reference answers form a set after normalization. Their order must not select the score.
    code = """
from lm_eval.tasks.libra import utils
results = [{"pred_answer": "красный кот", "answers": ["красный пёс", "красный кот"], "length": "8p"}]
assert utils.aggregate_results_f1(results) == {"8p": 1.0}
"""
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = hash_seed
    subprocess.run(  # noqa: S603 -- fixed interpreter and literal test script, no untrusted command input
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_libra_reference_count_score_uses_best_answer():
    results = [
        {"pred_answer": "1 2 2", "answers": ["1", "2"], "length": "8p"},
        {"pred_answer": "3", "answers": [], "length": "8p"},
    ]
    assert utils.aggregate_results_count_score(results) == {"8p": pytest.approx(1 / 3)}


def test_libra_exact_match_reference_aggregation_is_unchanged():
    results = [
        {"pred_answer": "красный кот", "answers": ["пёс", "кот"], "length": "8p"},
        {"pred_answer": "красный кот", "answers": ["пёс"], "length": "8p"},
    ]
    assert utils.aggregate_results_em(results) == {"8p": 0.5}
