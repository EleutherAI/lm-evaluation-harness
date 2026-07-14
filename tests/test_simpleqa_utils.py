"""
Tests for SimpleQA utils.py

Concept: Why test the scoring logic?
─────────────────────────────────────
The scoring function is the most important part of any eval task.
A buggy normalizer means you under-count correct answers
(your model looks dumber than it is) or over-count them
(it looks smarter). Either way, published numbers become misleading.

These tests verify:
  1. normalize_answer strips noise correctly
  2. token_f1 handles edge cases
  3. process_results classifies correct / incorrect / not_attempted
  4. The 'not_attempted' flag doesn't misfire on real answers
"""

import sys
import os

# Allow importing simpleqa utils from the task directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "lm_eval", "tasks", "simpleqa"))
from utils import normalize_answer, token_f1, is_not_attempted, process_results


# ─────────────────────────────────────────────────────────────
# normalize_answer
# ─────────────────────────────────────────────────────────────

class TestNormalizeAnswer:
    def test_lowercases(self):
        assert normalize_answer("Marie Curie") == "marie curie"

    def test_strips_punctuation(self):
        assert normalize_answer("1867.") == "1867"
        assert normalize_answer("Paris, France") == "paris france"

    def test_removes_leading_articles(self):
        assert normalize_answer("The Eiffel Tower") == "eiffel tower"
        assert normalize_answer("A river") == "river"
        assert normalize_answer("An apple") == "apple"

    def test_collapses_whitespace(self):
        assert normalize_answer("  Isaac   Newton  ") == "isaac newton"

    def test_empty_string(self):
        assert normalize_answer("") == ""

    def test_numbers_unchanged(self):
        assert normalize_answer("1867") == "1867"


# ─────────────────────────────────────────────────────────────
# token_f1
# Concept: F1 = harmonic mean of precision and recall on tokens
# ─────────────────────────────────────────────────────────────

class TestTokenF1:
    def test_perfect_match(self):
        assert token_f1("marie curie", "marie curie") == 1.0

    def test_no_overlap(self):
        assert token_f1("albert einstein", "marie curie") == 0.0

    def test_partial_overlap(self):
        # "new york city" vs "city of new york" — 3 tokens overlap out of 3+4
        f1 = token_f1("new york city", "city of new york")
        assert 0.0 < f1 < 1.0

    def test_both_empty(self):
        # Both empty → identical → 1.0
        assert token_f1("", "") == 1.0

    def test_one_empty(self):
        assert token_f1("", "something") == 0.0
        assert token_f1("something", "") == 0.0

    def test_subset_prediction(self):
        # "newton" vs "isaac newton" — partial
        f1 = token_f1("newton", "isaac newton")
        assert 0.0 < f1 < 1.0


# ─────────────────────────────────────────────────────────────
# is_not_attempted
# Concept: some "correct" words resemble refusals — we must not
# over-trigger the not_attempted detector.
# ─────────────────────────────────────────────────────────────

class TestIsNotAttempted:
    def test_explicit_refusals(self):
        assert is_not_attempted("I don't know") is True
        assert is_not_attempted("I'm not sure about this.") is True
        assert is_not_attempted("I cannot determine the answer.") is True
        assert is_not_attempted("I do not have enough information.") is True
        assert is_not_attempted("Cannot be determined from the given context.") is True

    def test_real_answers_not_flagged(self):
        # These look like answers, not refusals
        assert is_not_attempted("Marie Curie") is False
        assert is_not_attempted("1867") is False
        assert is_not_attempted("Paris, France") is False
        assert is_not_attempted("The answer is 42.") is False

    def test_edge_case_contains_know(self):
        # "know" appears in the answer but it's not a refusal
        assert is_not_attempted("Little Known Facts is a 1930 movie") is False


# ─────────────────────────────────────────────────────────────
# process_results — the full pipeline
# ─────────────────────────────────────────────────────────────

class TestProcessResults:
    def _make_doc(self, problem: str, answer: str) -> dict:
        return {
            "problem": problem,
            "answer": answer,
            "metadata": "{}",
        }

    def test_exact_match_passes(self):
        doc = self._make_doc("When was Marie Curie born?", "1867")
        result = process_results(doc, ["1867"])
        assert result["exact_match"] == 1.0
        assert result["f1"] == 1.0
        assert result["not_attempted"] == 0.0

    def test_case_insensitive_match(self):
        doc = self._make_doc("Who invented the telephone?", "Alexander Graham Bell")
        result = process_results(doc, ["alexander graham bell"])
        assert result["exact_match"] == 1.0

    def test_wrong_answer(self):
        doc = self._make_doc("What is the capital of France?", "Paris")
        result = process_results(doc, ["London"])
        assert result["exact_match"] == 0.0
        assert result["f1"] == 0.0
        assert result["not_attempted"] == 0.0

    def test_not_attempted_detected(self):
        doc = self._make_doc("What is the boiling point of helium?", "-269 degrees Celsius")
        result = process_results(doc, ["I don't know the answer to this question."])
        assert result["exact_match"] == 0.0
        assert result["f1"] == 0.0
        assert result["not_attempted"] == 1.0

    def test_partial_credit_via_f1(self):
        # Model says "Graham Bell" instead of "Alexander Graham Bell"
        doc = self._make_doc("Who invented the telephone?", "Alexander Graham Bell")
        result = process_results(doc, ["Graham Bell"])
        assert result["exact_match"] == 0.0   # not a full match
        assert result["f1"] > 0.0             # but some token overlap

    def test_article_stripped_match(self):
        # Reference has "the" — normalizer should strip it
        doc = self._make_doc("What river runs through London?", "The Thames")
        result = process_results(doc, ["Thames"])
        assert result["exact_match"] == 1.0

    def test_trailing_punctuation_stripped(self):
        doc = self._make_doc("What year did WWII end?", "1945")
        result = process_results(doc, ["1945."])
        assert result["exact_match"] == 1.0

    def test_results_must_have_one_item(self):
        import pytest
        doc = self._make_doc("Q", "A")
        with pytest.raises(AssertionError):
            process_results(doc, ["answer1", "answer2"])

    def test_whitespace_padded_output(self):
        doc = self._make_doc("Capital of Japan?", "Tokyo")
        result = process_results(doc, ["  Tokyo  "])
        assert result["exact_match"] == 1.0
