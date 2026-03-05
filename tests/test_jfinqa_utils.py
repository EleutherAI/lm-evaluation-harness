"""Tests for jfinqa task utilities."""

import pytest

from lm_eval.tasks.jfinqa.utils import (
    NUMERICAL_TOLERANCE,
    _extract_answer,
    _normalize,
    _numerical_match,
    _try_parse_number,
    doc_to_text,
    process_results,
)


class TestNormalize:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("  hello  ", "hello"),
            ("HELLO", "hello"),
            ("１２３", "123"),  # Full-width -> half-width (NFKC)
            ("△10.5%", "-10.5%"),
            ("▲20%", "-20%"),
            ("1,234,567", "1234567"),
            ("増加しました", "増加"),
            ("改善した", "改善"),
            ("悪化", "悪化"),  # No suffix to remove
            ("", ""),
        ],
    )
    def test_normalize(self, text, expected):
        assert _normalize(text) == expected

    def test_normalize_comma_only_between_digits(self):
        assert _normalize("a,b") == "a,b"  # Comma not between digits
        assert _normalize("1,000") == "1000"  # Comma between digits


class TestExtractAnswer:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("Answer: 42%", "42%"),
            ("answer: 100百万円", "100百万円"),
            ("A: はい", "はい"),
            ("Answer：50%", "50%"),  # Full-width colon
            ("Some reasoning\nAnswer: 10.5%", "10.5%"),
            ("回答: 42%", "42%"),  # Japanese "Answer:"
            ("回答：100百万円", "100百万円"),  # Japanese with full-width colon
            ("line one\nline two\nline three", "line three"),  # Fallback to last line
            ("", ""),
        ],
    )
    def test_extract_answer(self, text, expected):
        assert _extract_answer(text) == expected

    def test_extract_answer_multiline_with_answer(self):
        text = "Let me think step by step.\nThe revenue is 100.\nAnswer: 100百万円"
        assert _extract_answer(text) == "100百万円"


class TestTryParseNumber:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("42", 42.0),
            ("-10.5", -10.5),
            ("10.5%", 10.5),
            ("100百万円", 100.0),  # Unit suffix stripped as label
            ("5千円", 5.0),
            ("3億円", 3.0),
            ("2兆円", 2.0),
            ("1,234百万円", 1234.0),
            ("50ドル", 50.0),
            ("3.5ポイント", 3.5),
            ("100bps", 100.0),
            ("△5.2%", -5.2),  # Triangle normalized to minus sign
        ],
    )
    def test_parse_number(self, text, expected):
        result = _try_parse_number(text)
        assert result is not None
        assert abs(result - expected) < 1e-6

    def test_parse_unparseable(self):
        assert _try_parse_number("はい") is None
        assert _try_parse_number("改善") is None

    def test_parse_negative(self):
        result = _try_parse_number("-30.6%")
        assert result is not None
        assert abs(result - (-30.6)) < 1e-6


class TestNumericalMatch:
    def test_tolerance_constant(self):
        assert NUMERICAL_TOLERANCE == 0.01

    def test_exact_numerical_match(self):
        assert _numerical_match("10.5%", "10.5%") is True

    def test_within_tolerance(self):
        # 10.5 vs 10.4 -> |0.1/10.4| ≈ 0.0096 < 0.01
        assert _numerical_match("10.5%", "10.4%") is True

    def test_outside_tolerance(self):
        # 10.5 vs 10.0 -> |0.5/10.0| = 0.05 > 0.01
        assert _numerical_match("10.5%", "10.0%") is False

    def test_zero_gold(self):
        assert _numerical_match("0", "0") is True
        assert _numerical_match("0.1", "0") is False

    def test_non_numeric_fallback(self):
        assert _numerical_match("はい", "はい") is True
        assert _numerical_match("はい", "いいえ") is False

    def test_unit_match(self):
        # 100百万円 vs 100百万円
        assert _numerical_match("100百万円", "100百万円") is True

    def test_same_unit_different_values(self):
        # Same unit, values differ by more than 1%
        assert _numerical_match("100百万円", "200百万円") is False


class TestDocToText:
    def test_complete_document(self):
        doc = {
            "pre_text": ["Revenue increased."],
            "table_headers": ["Item", "2023", "2024"],
            "table_rows": [["Revenue", "100", "120"]],
            "post_text": ["Note: in millions."],
            "question": "What is the growth rate?",
        }
        result = doc_to_text(doc)
        assert "Revenue increased." in result
        assert "| Item | 2023 | 2024 |" in result
        assert "| Revenue | 100 | 120 |" in result
        assert "Note: in millions." in result
        assert "Question: What is the growth rate?" in result
        assert result.endswith("Answer:")

    def test_missing_optional_fields(self):
        doc = {"question": "Is this correct?"}
        result = doc_to_text(doc)
        assert "Question: Is this correct?" in result
        assert result.endswith("Answer:")

    def test_no_table(self):
        doc = {
            "pre_text": ["Some text."],
            "table_headers": [],
            "table_rows": [],
            "question": "What happened?",
        }
        result = doc_to_text(doc)
        assert "| " not in result or "| ---" not in result


class TestProcessResults:
    def test_exact_and_numerical_match(self):
        doc = {"answer": "10.5%"}
        result = process_results(doc, ["Answer: 10.5%"])
        assert result["exact_match"] == 1.0
        assert result["numerical_match"] == 1.0

    def test_numerical_match_only(self):
        doc = {"answer": "10.4%"}
        result = process_results(doc, ["Answer: 10.5%"])
        assert result["exact_match"] == 0.0
        assert result["numerical_match"] == 1.0  # Within 1% tolerance

    def test_no_match(self):
        doc = {"answer": "50.0%"}
        result = process_results(doc, ["Answer: 10.5%"])
        assert result["exact_match"] == 0.0
        assert result["numerical_match"] == 0.0

    def test_empty_results(self):
        doc = {"answer": "10.5%"}
        result = process_results(doc, [])
        assert result["exact_match"] == 0.0

    def test_japanese_text_match(self):
        doc = {"answer": "改善"}
        result = process_results(doc, ["Answer: 改善した"])
        assert result["exact_match"] == 1.0  # "した" suffix removed by _normalize
