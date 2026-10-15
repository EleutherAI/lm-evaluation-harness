"""Lexical substitution covers each annotated synonym at most once."""

import pytest

from lm_eval.tasks.evalita_llm.metrics import _aggreg_ls
from lm_eval.tasks.evalita_llm.utils import ls_process_results


@pytest.mark.parametrize(
    "answer, expected",
    [
        ("bello", 2 / 3),
        ("bello, bello", 2 / 3),
        ("bello, buono", 1.0),
        ("bello, buono, bello, buono", 1.0),
        ("inesistente, inesistente", 0.0),
        (", ".join(["bello"] * 10 + ["buono"]), 2 / 3),
    ],
)
def test_synonym_frequency_mass_cannot_be_counted_twice(answer, expected):
    doc = {"answers": [{"word": "bello", "count": 2}, {"word": "buono", "count": 1}]}
    result = ls_process_results(doc, [answer])["f1"]
    assert result == pytest.approx((expected, 1, 1))
    assert _aggreg_ls([result]) == pytest.approx(expected)


def test_repeating_a_single_gold_synonym_cannot_make_f1_exceed_one():
    doc = {"answers": [{"word": "bello", "count": 1}]}
    repeated = ls_process_results(doc, [", ".join(["bello"] * 10)])["f1"]
    assert repeated == (1.0, 1, 1)
    assert _aggreg_ls([repeated]) == 1.0


def test_corpus_score_agrees_with_unique_gold_frequency_coverage():
    doc = {"answers": [{"word": "bello", "count": 2}, {"word": "buono", "count": 1}]}
    items = [
        ls_process_results(doc, [answer])["f1"]
        for answer in ["bello, bello", "bello, buono", "inesistente"]
    ]
    # Sum of covered annotator mass is 2/3 + 1 + 0 across three answered items.
    assert _aggreg_ls(items) == pytest.approx(5 / 9)
