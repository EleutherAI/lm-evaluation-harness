"""
Tests for SimpleQA scoring utilities.

Run with:  pytest tests/simpleqa/test_simpleqa.py -v
or from the repo root: python -m pytest lm_eval/tasks/simpleqa/test_simpleqa.py -v
"""
import pytest

# When running from the repo root the import path is:
#   from lm_eval.tasks.simpleqa.utils import ...
# For standalone testing adjust as needed.
from utils import (
    classify_answer,
    normalize_answer,
    process_results,
    simpleqa_f1_agg,
)


# ---------------------------------------------------------------------------
# normalize_answer
# ---------------------------------------------------------------------------

class TestNormalizeAnswer:
    def test_lowercase(self):
        assert normalize_answer("Barack Obama") == "barack obama"

    def test_removes_articles(self):
        assert normalize_answer("The White House") == "white house"
        assert normalize_answer("a cat sat on an mat") == "cat sat on mat"

    def test_removes_punctuation(self):
        assert normalize_answer("1,200") == "1200"
        assert normalize_answer("San Francisco, California") == "san francisco california"

    def test_collapses_whitespace(self):
        assert normalize_answer("  hello   world  ") == "hello world"

    def test_empty_string(self):
        assert normalize_answer("") == ""


# ---------------------------------------------------------------------------
# classify_answer — CORRECT cases
# ---------------------------------------------------------------------------

class TestClassifyCorrect:
    """
    These cases mirror the CORRECT examples from the original SimpleQA grader
    prompt and from the paper's grading guidelines.
    """

    def test_exact_match(self):
        assert classify_answer("Malia and Sasha", "Malia and Sasha") == "correct"

    def test_case_insensitive(self):
        assert classify_answer("malia and sasha obama", "Malia Obama and Sasha Obama") == "correct"

    def test_prediction_subset_of_gold(self):
        # "Michelle" is correct when gold is "Michelle Obama"
        # because the last name is implied by the question.
        assert classify_answer("Michelle", "Michelle Obama") == "correct"

    def test_prediction_superset_of_gold(self):
        # Model says "San Francisco, California"; gold is "San Francisco"
        assert classify_answer("San Francisco, California", "San Francisco") == "correct"

    def test_unit_in_gold_not_prediction(self):
        # Gold "1.73 m"; prediction "1.73" — unit implied in the question
        assert classify_answer("1.73", "1.73 m") == "correct"

    def test_hedged_but_correct(self):
        # Hedging is OK if the gold is present (original paper rule)
        assert classify_answer(
            "I believe the answer is Malia and Sasha Obama",
            "Malia and Sasha",
        ) == "correct"

    def test_award_suffix_omitted(self):
        # "Outstanding Paper" vs "Outstanding Paper Award" — award implied in question
        assert classify_answer("Outstanding Paper", "Outstanding Paper Award") == "correct"

    def test_article_difference(self):
        assert classify_answer("White House", "The White House") == "correct"

    def test_refusal_phrase_but_answer_present(self):
        # Unusual but should still be CORRECT if gold appears in output
        assert classify_answer(
            "I'm not sure, but I think it's Sasha and Malia Obama",
            "Malia and Sasha",
        ) == "correct"


# ---------------------------------------------------------------------------
# classify_answer — INCORRECT cases
# ---------------------------------------------------------------------------

class TestClassifyIncorrect:
    def test_wrong_name(self):
        assert classify_answer("Malia.", "Malia and Sasha") == "incorrect"

    def test_extra_wrong_name(self):
        assert classify_answer("Malia, Sasha, and Susan", "Malia and Sasha") == "incorrect"

    def test_negation(self):
        assert classify_answer(
            "Barack Obama does not have any children.",
            "Malia and Sasha",
        ) == "incorrect"

    def test_wrong_count(self):
        assert classify_answer(
            "Barack Obama has three children.",
            "Malia and Sasha",
        ) == "incorrect"

    def test_completely_wrong(self):
        assert classify_answer("Paris", "London") == "incorrect"


# ---------------------------------------------------------------------------
# classify_answer — NOT_ATTEMPTED cases
# ---------------------------------------------------------------------------

class TestClassifyNotAttempted:
    def test_i_dont_know(self):
        assert classify_answer("I don't know.", "Malia and Sasha") == "not_attempted"

    def test_i_do_not_know(self):
        assert classify_answer("I do not know.", "Malia and Sasha") == "not_attempted"

    def test_need_more_context(self):
        assert classify_answer(
            "I need more context about which Obama you are talking about.",
            "Malia and Sasha",
        ) == "not_attempted"

    def test_cannot_answer_without_web(self):
        assert classify_answer(
            "Without researching the web, I cannot answer this question.",
            "Malia and Sasha",
        ) == "not_attempted"

    def test_partial_then_unsure(self):
        # "I know one but not the other" → NOT_ATTEMPTED (gold not fully present)
        assert classify_answer(
            "Barack Obama has two children. I know that one of them is Malia, "
            "but I'm not sure about the other one.",
            "Malia and Sasha",
        ) == "not_attempted"

    def test_i_cannot(self):
        assert classify_answer("I cannot answer this.", "42") == "not_attempted"

    def test_i_am_unable(self):
        assert classify_answer("I am unable to provide that information.", "42") == "not_attempted"


# ---------------------------------------------------------------------------
# process_results
# ---------------------------------------------------------------------------

class TestProcessResults:
    def _doc(self, answer):
        return {"problem": "...", "answer": answer}

    def test_correct_returns_correct_1(self):
        result = process_results(self._doc("Malia and Sasha"), ["Malia and Sasha Obama"])
        assert result["correct"] == 1
        assert result["not_attempted"] == 0
        assert result["f1"] == (1, 0, 0)

    def test_incorrect_returns_correct_0(self):
        result = process_results(self._doc("Paris"), ["London"])
        assert result["correct"] == 0
        assert result["not_attempted"] == 0
        assert result["f1"] == (0, 1, 0)

    def test_not_attempted(self):
        result = process_results(self._doc("Malia and Sasha"), ["I don't know."])
        assert result["correct"] == 0
        assert result["not_attempted"] == 1
        assert result["f1"] == (0, 0, 1)


# ---------------------------------------------------------------------------
# simpleqa_f1_agg
# ---------------------------------------------------------------------------

class TestSimpleQAF1Agg:
    def test_all_correct(self):
        # accuracy_given_attempted = 1.0, overall = 1.0  → F1 = 1.0
        items = [(1, 0, 0)] * 10
        assert simpleqa_f1_agg(items) == pytest.approx(1.0)

    def test_all_incorrect(self):
        # accuracy_given_attempted = 0.0 → F1 = 0.0
        items = [(0, 1, 0)] * 10
        assert simpleqa_f1_agg(items) == pytest.approx(0.0)

    def test_all_not_attempted(self):
        # No attempts → F1 = 0.0
        items = [(0, 0, 1)] * 10
        assert simpleqa_f1_agg(items) == pytest.approx(0.0)

    def test_half_correct_half_not_attempted(self):
        # correct=5, incorrect=0, not_attempted=5, total=10
        # acc_given_attempted = 5/5 = 1.0
        # overall_correct = 5/10 = 0.5
        # F1 = 2 * 1.0 * 0.5 / (1.0 + 0.5) = 1.0 / 1.5 ≈ 0.6667
        items = [(1, 0, 0)] * 5 + [(0, 0, 1)] * 5
        assert simpleqa_f1_agg(items) == pytest.approx(2 / 3, rel=1e-4)

    def test_half_correct_half_incorrect(self):
        # correct=5, incorrect=5, not_attempted=0, total=10
        # acc_given_attempted = 5/10 = 0.5
        # overall_correct = 5/10 = 0.5
        # F1 = 2 * 0.5 * 0.5 / (0.5 + 0.5) = 0.5
        items = [(1, 0, 0)] * 5 + [(0, 1, 0)] * 5
        assert simpleqa_f1_agg(items) == pytest.approx(0.5)

    def test_empty(self):
        assert simpleqa_f1_agg([]) == pytest.approx(0.0)

    def test_realistic_scores(self):
        # Simulate a model with ~40% correct, ~10% not_attempted, ~50% incorrect
        # (roughly what a mid-tier model might score)
        n = 100
        items = (
            [(1, 0, 0)] * 40
            + [(0, 1, 0)] * 50
            + [(0, 0, 1)] * 10
        )
        f1 = simpleqa_f1_agg(items)
        # acc_given_attempted = 40/90 ≈ 0.444
        # overall_correct = 40/100 = 0.400
        # F1 = 2 * 0.444 * 0.4 / (0.444 + 0.4) ≈ 0.421
        assert 0.40 < f1 < 0.45
