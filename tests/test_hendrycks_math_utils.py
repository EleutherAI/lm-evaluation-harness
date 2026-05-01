"""Tests for Hendrycks Math answer extraction (utils.py).

Regression for #2552: the original extraction only handled inline $...$
delimiters; model responses using display math \\[...\\] were silently
treated as having no delimiter, causing the full verbatim response string
to be compared against the ground-truth boxed answer — almost always wrong.
"""

import unittest

from lm_eval.tasks.hendrycks_math.utils import _extract_answer, process_results


class TestExtractAnswer(unittest.TestCase):
    # ------------------------------------------------------------------
    # Original inline-math behaviour preserved
    # ------------------------------------------------------------------

    def test_inline_math_simple(self):
        self.assertEqual(_extract_answer("$42$"), "42")

    def test_inline_math_with_surrounding_text(self):
        result = _extract_answer("The answer is $x = 5$.")
        self.assertEqual(result, "x = 5")

    def test_inline_math_uses_outermost_delimiters(self):
        # Multiple $ pairs — should span first to last
        result = _extract_answer("We have $a$ and $b = 3$")
        self.assertEqual(result, "a$ and $b = 3")

    def test_single_dollar_sign_no_extraction(self):
        # Only one $ — fall back to full string
        result = _extract_answer("answer is $42 dollars")
        self.assertEqual(result, "answer is $42 dollars")

    # ------------------------------------------------------------------
    # New display-math \\[...\\] behaviour (#2552)
    # ------------------------------------------------------------------

    def test_display_math_simple(self):
        result = _extract_answer("\\[42\\]")
        self.assertEqual(result, "42")

    def test_display_math_with_boxed(self):
        result = _extract_answer("Thus the answer is \\[ \\boxed{42} \\]")
        self.assertEqual(result, "\\boxed{42}")

    def test_display_math_preferred_over_inline_when_both_present(self):
        # \\[...\\] should win because it is checked first
        result = _extract_answer("So $x=1$ but the final answer is \\[ 2 \\]")
        self.assertEqual(result, "2")

    def test_display_math_multiline_answer(self):
        result = _extract_answer("We get \\[ \\frac{1}{2} \\]")
        self.assertEqual(result, "\\frac{1}{2}")

    # ------------------------------------------------------------------
    # No-delimiter fallback
    # ------------------------------------------------------------------

    def test_no_delimiter_returns_full_string(self):
        result = _extract_answer("42")
        self.assertEqual(result, "42")

    def test_empty_string(self):
        result = _extract_answer("")
        self.assertEqual(result, "")


class TestProcessResults(unittest.TestCase):
    """Integration tests for process_results using the corrected extractor."""

    def _doc(self, solution):
        return {"solution": solution}

    def test_inline_math_correct(self):
        doc = self._doc("... \\boxed{42}")
        result = process_results(doc, ["The answer is $42$."])
        self.assertEqual(result["exact_match"], 1)

    def test_display_math_correct(self):
        """Regression for #2552: display math was never matched before this fix."""
        doc = self._doc("... \\boxed{42}")
        result = process_results(doc, ["Thus the answer is \\[ \\boxed{42} \\]"])
        self.assertEqual(result["exact_match"], 1)

    def test_wrong_answer_no_match(self):
        doc = self._doc("... \\boxed{42}")
        result = process_results(doc, ["The answer is $7$."])
        self.assertEqual(result["exact_match"], 0)

    def test_display_math_wrong_answer(self):
        doc = self._doc("... \\boxed{42}")
        result = process_results(doc, ["\\[ \\boxed{7} \\]"])
        self.assertEqual(result["exact_match"], 0)


if __name__ == "__main__":
    unittest.main()
