import pytest

from lm_eval.tasks.quac.utils import _f1_score, process_results


def test_f1_normalizes_case_articles_and_punctuation():
    assert _f1_score("The Red, Balloon!", "red balloon") == 1.0


def test_f1_counts_duplicate_tokens():
    assert _f1_score("red red blue", "red blue blue") == pytest.approx(2 / 3)


def test_process_results_uses_best_reference_and_first_line():
    doc = {"references": ["CANNOTANSWER", "Not enough information"]}

    assert process_results(doc, ["Not enough information\nQuestion: next"]) == {
        "f1": 1.0
    }


def test_process_results_returns_zero_without_overlap():
    doc = {"references": ["one answer", "another answer"]}

    assert process_results(doc, ["unrelated output"]) == {"f1": 0.0}
