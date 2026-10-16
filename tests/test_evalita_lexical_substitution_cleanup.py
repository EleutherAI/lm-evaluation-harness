"""Lexical-substitution cleanup preserves valid gold punctuation."""

import pytest

from lm_eval.tasks.evalita_llm.utils import ls_process_results


@pytest.mark.parametrize(
    ("gold", "answer"),
    [
        ("gruppo dirigente", "gruppo dirigente"),
        ("realta'", "realta'"),
        ("auto?", "auto?"),
        ("felice", "felice."),
        ("felice", "felice!"),
        ("felice", "felice;"),
        ("felice", "felice:"),
        ("felice", "contento, felice."),
    ],
)
def test_last_substitute_cleanup_preserves_gold_text(gold, answer):
    result = ls_process_results({"answers": [{"word": gold, "count": 1}]}, [answer])["f1"]
    assert result == pytest.approx((1.0, 1, 1))
