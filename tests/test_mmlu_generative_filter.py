"""Tests for the mmlu_generative get_response filter chain.

Regression for issue #2279: the chain previously kept the whole first
line, so any answer format other than a bare letter ("A.", "(A)",
"Answer: A", "The answer is B") scored exact_match=0 even when the
model's choice was correct. The chain now extracts the choice letter
from decorated formats; lines without an isolated A-D letter yield the
regex fallback and score 0 exactly as before.
"""

import pytest

from lm_eval.filters.extraction import WhitespaceFilter
from lm_eval.filters.extraction import RegexFilter
from lm_eval.filters.selection import TakeFirstFilter


def build_chain():
    """The exact get_response chain from
    lm_eval/tasks/mmlu/generative/_default_template_yaml, kept in sync."""
    return [
        RegexFilter(regex_pattern=r"^(.*?)(?=\n|$)"),
        WhitespaceFilter(),
        RegexFilter(regex_pattern=r"^(.*?)\s*$"),
        RegexFilter(
            regex_pattern=(
                r"(?i)(?:^\W*|answer\s*(?:is)?\s*[:\-]?\s*|option\s*[:\-]?\s*)"
                r"\(?([A-D])\)?\W*$"
            )
        ),
        TakeFirstFilter(),
    ]


def apply_chain(resps):
    out = resps
    for filt in build_chain():
        out = list(filt.apply(out, docs=[{}] * len(out)))
    return out


class TestLetterExtraction:
    @pytest.mark.parametrize(
        ("response", "letter"),
        [
            ("A", "A"),
            ("a", "a"),
            ("A.", "A"),
            ("A)", "A"),
            ("(A)", "A"),
            ("a)", "a"),
            ("Answer: A", "A"),
            ("answer: a", "a"),
            ("Answer is B", "B"),
            ("answer - C", "C"),
            ("The answer is B", "B"),
            ("option: D", "D"),
            (".D", "D"),
        ],
    )
    def test_decorated_formats_extract(self, response, letter):
        # the final take_first step yields a bare string per document
        assert apply_chain([[response]]) == [letter]

    @pytest.mark.parametrize(
        "response",
        [
            "A. foo",
            "Because Paris",
            "AB",
            "A B",
            "The answer is Paris",
            "",
        ],
    )
    def test_ambiguous_lines_fall_back(self, response):
        # no isolated A-D letter: the regex fallback fires and the item
        # scores 0 against a bare-letter target, exactly as before the fix
        out = apply_chain([[response]])
        assert out == ["[invalid]"]

    def test_multiline_takes_first_line(self):
        assert apply_chain([["Answer: B\nbecause reasons"]]) == ["B"]

    def test_bare_letter_unchanged(self):
        # the pre-fix behavior for well-formed answers is byte-identical
        assert apply_chain([["A"], ["b"], ["D"]]) == ["A", "b", "D"]


class TestScoreEquivalence:
    """Every line the new regex does not match was already scoring 0
    under whole-line comparison against a bare-letter target."""

    @pytest.mark.parametrize("response", ["A. foo", "Because Paris", "", "AB"])
    def test_fallback_scores_zero_like_before(self, response):
        from lm_eval.api.metrics import exact_match_hf_evaluate

        filtered = apply_chain([[response]])[0]
        score = exact_match_hf_evaluate(
            predictions=[filtered],
            references=["A"],
            ignore_punctuation=True,
            ignore_case=True,
        )["exact_match"]
        whole_line = exact_match_hf_evaluate(
            predictions=[response.strip()],
            references=["A"],
            ignore_punctuation=True,
            ignore_case=True,
        )["exact_match"]
        assert score == whole_line == 0.0


class TestYamlSync:
    def test_template_contains_extraction_step(self):
        import pathlib

        template = (
            pathlib.Path(__file__).parent.parent
            / "lm_eval/tasks/mmlu/generative/_default_template_yaml"
        ).read_text()
        # the yaml file stores the pattern with escaped backslashes
        assert "(?i)(?:^\\\\W*|answer" in template
        # the extraction step sits after the trim step, before take_first
        i_trim = template.index("^(.*?)\\\\s*$")
        i_extract = template.index("(?i)(?:^\\\\W*|answer")
        i_take = template.index("- function: take_first")
        assert i_trim < i_extract < i_take
