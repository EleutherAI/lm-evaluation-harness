"""Tests for the Formats configuration system.

Tests cover:
- FormatConfig registry and __init_subclass__ auto-registration
- FormatConfig.get() resolution from str, dict, instance, None
- Jinja template generation (doc_to_text, doc_to_target, doc_to_choice)
- Jinja template rendering with sample documents
- to_task_config() output
- All built-in formats: MCQFormat, ClozeFormat, GenerateFormat, COTGenerateFormat
- TaskConfig._resolve_formats() integration (@ suffix, formats field, overrides)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pytest
from jinja2 import Environment

from lm_eval.config.formats import (
    ClozeFormat,
    COTGenFormat,
    FormatConfig,
    GenerateFormat,
    MCQAFormat,
)
from lm_eval.config.task import TaskConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

jinja_env = Environment()


def render(template_str: str, doc: dict) -> str:
    """Render a Jinja2 template string with a document dict."""
    return jinja_env.from_string(template_str).render(**doc)


# A typical MCQ document (e.g. MMLU-style)
MCQ_DOC = {
    "question": "What is the capital of France?",
    "choices": ["Berlin", "Madrid", "Paris", "Rome"],
    "answer": 2,  # index into choices -> "Paris"
}

# A generation-style document (free-form answer)
GEN_DOC = {
    "question": "What is 2 + 2?",
    "answer": "4",
}

# A document with a text answer instead of an index
TEXT_ANSWER_DOC = {
    "question": "Which planet is closest to the Sun?",
    "choices": ["Venus", "Mercury", "Mars", "Jupiter"],
    "answer": "Mercury",
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    """Test __init_subclass__ auto-registration and list_formatss()."""

    def test_builtin_formatss_registered(self):
        registered = FormatConfig.registered()
        assert "mcqa" in registered
        assert "cloze" in registered
        assert "generate" in registered
        assert "cot" in registered

    def test_list_formatss_sorted(self):
        registered = FormatConfig.registered()
        assert registered == sorted(registered)

    def test_custom_subclass_registers(self):
        """A new subclass with format_name auto-registers."""

        @dataclass(kw_only=True)
        class _TestFormat(MCQAFormat):
            format_name: ClassVar[str | None] = "_test_only_formats"

        assert "_test_only_formats" in FormatConfig._registry
        assert FormatConfig._registry["_test_only_formats"] is _TestFormat

        # Cleanup to avoid polluting other tests
        del FormatConfig._registry["_test_only_formats"]

    def test_subclass_without_format_name_not_registered(self):
        """A subclass that leaves format_name = None is NOT registered."""

        @dataclass(kw_only=True)
        class _InternalFormat(MCQAFormat):
            format_name: ClassVar[str | None] = None

        assert "_InternalFormat" not in FormatConfig._registry
        # None is already a key? It shouldn't be.
        for name in FormatConfig._registry:
            assert FormatConfig._registry[name] is not _InternalFormat


# ---------------------------------------------------------------------------
# FormatConfig.get()
# ---------------------------------------------------------------------------


class TestGet:
    """Test the FormatConfig.get() resolution logic."""

    def test_get_none_returns_none(self):
        assert FormatConfig.get(None) is None

    def test_get_instance_returns_same(self):
        formats = MCQAFormat()
        assert FormatConfig.get(formats) is formats

    def test_get_string_returns_instance(self):
        formats = FormatConfig.get("mcqa")
        assert isinstance(formats, MCQAFormat)

    def test_get_string_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown format"):
            FormatConfig.get("nonexistent_formats")

    def test_get_dict_with_type_key(self):
        formats = FormatConfig.get(
            {"type": "mcqa", "instruction": "Custom instruction"}
        )
        assert isinstance(formats, MCQAFormat)
        assert formats.instruction == "Custom instruction"

    def test_get_dict_with_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown format type"):
            FormatConfig.get({"type": "nonexistent"})

    def test_get_multi_formats_dict_default_first(self):
        spec = {
            "mcqa": {"instruction": "MCQ instruction"},
            "generate": {"instruction": "Gen instruction"},
        }
        formats = FormatConfig.get(spec)
        assert isinstance(formats, MCQAFormat)
        assert formats.instruction == "MCQ instruction"

    def test_get_multi_formats_dict_with_selection(self):
        spec = {
            "mcqa": {"instruction": "MCQ instruction"},
            "generate": {"instruction": "Gen instruction"},
        }
        formats = FormatConfig.get(spec, selection="generate")
        assert isinstance(formats, GenerateFormat)
        assert formats.instruction == "Gen instruction"

    def test_get_multi_formats_dict_selection_not_found(self):
        spec = {"mcqa": None}
        with pytest.raises(ValueError, match="not found in task formats"):
            FormatConfig.get(spec, selection="generate")

    def test_get_multi_formats_dict_none_overrides(self):
        """When the overrides value is None, use default formats."""
        spec = {"mcqa": None}
        formats = FormatConfig.get(spec)
        assert isinstance(formats, MCQAFormat)
        # Should have default MCQFormat values
        assert formats.question_prefix == "Question: "

    def test_get_multi_formats_dict_string_redirect(self):
        """When an override value is a string, redirect to that formats."""
        spec = {"mcqa": "generate"}
        formats = FormatConfig.get(spec)
        assert isinstance(formats, GenerateFormat)

    def test_get_invalid_type_raises(self):
        with pytest.raises(TypeError, match="Invalid format spec type"):
            FormatConfig.get(42)  # type: ignore[arg-type]

    def test_get_empty_string_raises(self):
        """Empty string is not a valid formats name."""
        with pytest.raises(ValueError, match="Unknown format"):
            FormatConfig.get("")

    def test_get_redirect_to_unknown_raises(self):
        """Multi-formats string redirect to unknown formats raises."""
        with pytest.raises(ValueError, match="Unknown format"):
            FormatConfig.get({"mcqa": "nonexistent_xyz"})


# ---------------------------------------------------------------------------
# _field_ref
# ---------------------------------------------------------------------------


class TestFieldRef:
    """Test the _field_ref helper method."""

    def setup_method(self):
        self.formats = MCQAFormat()

    def test_plain_field_for_output(self):
        assert self.formats._field_ref("question", for_output=True) == "{{question}}"

    def test_plain_field_for_control(self):
        assert self.formats._field_ref("question", for_output=False) == "question"

    def test_jinja_expression_for_output(self):
        expr = "{{doc['question']}}"
        assert self.formats._field_ref(expr, for_output=True) == "{{doc['question']}}"

    def test_jinja_expression_for_control(self):
        expr = "{{doc['question']}}"
        result = self.formats._field_ref(expr, for_output=False)
        assert "{{" not in result
        assert "}}" not in result
        assert "doc['question']" in result

    def test_complex_expression_parenthesized(self):
        """Complex expressions are wrapped in () for safe |filter and .method() usage."""
        result = self.formats._field_ref("{{[a, b, c]}}", for_output=False)
        assert result == "([a, b, c])"

    def test_simple_wrapped_field_not_parenthesized(self):
        """A simple identifier wrapped in {{ }} stays plain (no parens needed)."""
        result = self.formats._field_ref("{{choices}}", for_output=False)
        assert result == "choices"

    def test_dotted_access_parenthesized(self):
        """Dotted field access gets parenthesized for safe filter chaining."""
        result = self.formats._field_ref("{{choices.text}}", for_output=False)
        assert result == "(choices.text)"

    def test_filter_expression_parenthesized(self):
        """Piped filter expressions get parenthesized."""
        result = self.formats._field_ref(
            "{{answers|map(attribute='a')|list}}", for_output=False
        )
        assert result.startswith("(") and result.endswith(")")


# ---------------------------------------------------------------------------
# MCQFormat
# ---------------------------------------------------------------------------


class TestMCQFormat:
    """Test MCQFormat defaults and Jinja generation."""

    def setup_method(self):
        self.formats = MCQAFormat()

    def test_defaults(self):
        assert self.formats.output_type == "multiple_choice"
        assert self.formats.instruction is None
        assert self.formats.question_prefix == "Question: "
        assert self.formats.choice_labels == "letters"
        assert self.formats.answer_prompt == "Answer:"
        assert self.formats.scorer is None

    def test_jinja_doc_to_text_renders(self):
        cfg = self.formats.to_jinja_config()
        rendered = render(cfg["doc_to_text"], MCQ_DOC)
        # Should contain the question text
        assert "What is the capital of France?" in rendered
        # Should contain lettered choices
        assert "A." in rendered
        assert "B." in rendered
        assert "C." in rendered
        assert "D." in rendered
        # Should contain the choice text
        assert "Berlin" in rendered
        assert "Paris" in rendered
        # Should end with "Answer:"
        assert rendered.rstrip().endswith("Answer:")
        # Should have "Question:" prefix
        assert "Question:" in rendered

    def test_jinja_doc_to_target_renders_index(self):
        """MCQ target with numeric answer returns index directly."""
        cfg = self.formats.to_jinja_config()
        rendered = render(cfg["doc_to_target"], MCQ_DOC)
        assert rendered.strip() == "2"

    def test_jinja_doc_to_target_renders_text_answer(self):
        """MCQ target with text answer looks up index in choices."""
        cfg = self.formats.to_jinja_config()
        rendered = render(cfg["doc_to_target"], TEXT_ANSWER_DOC)
        assert rendered.strip() == "1"  # "Mercury" is at index 1

    def test_jinja_doc_to_choice_renders_labels(self):
        cfg = self.formats.to_jinja_config()
        rendered = render(cfg["doc_to_choice"], MCQ_DOC)  # type:ignore[invalid-argument-type]
        # Should produce letter labels: ['A', 'B', 'C', 'D']
        assert "A" in rendered
        assert "D" in rendered

    def test_to_task_config_fields(self):
        cfg = self.formats.to_task_config()
        assert cfg["output_type"] == "multiple_choice"
        assert "doc_to_text" in cfg
        assert "doc_to_target" in cfg
        assert "doc_to_choice" in cfg
        assert cfg["target_delimiter"] == " "
        assert cfg["fewshot_delimiter"] == "\n\n"
        # MCQ has no gen_prefix by default
        assert "gen_prefix" not in cfg

    def test_no_instruction(self):
        """MCQ default has no instruction."""
        cfg = self.formats.to_jinja_config()
        rendered = render(cfg["doc_to_text"], MCQ_DOC)
        # Should start with Question:, not an instruction
        assert rendered.lstrip().startswith("Question:")


# ---------------------------------------------------------------------------
# ClozeFormat
# ---------------------------------------------------------------------------


class TestClozeFormat:
    """Test ClozeFormat (MCQ variant without choice labels)."""

    def setup_method(self):
        self.formats = ClozeFormat()

    def test_inherits_mcq(self):
        assert isinstance(self.formats, MCQAFormat)
        assert self.formats.output_type == "multiple_choice"

    def test_no_choice_labels(self):
        assert self.formats.choice_labels is None

    def test_jinja_doc_to_text_no_labels(self):
        cfg = self.formats.to_jinja_config()
        rendered = render(cfg["doc_to_text"], MCQ_DOC)
        # Should NOT contain "A." "B." style labels
        assert "A." not in rendered
        # Should still contain the question
        assert "What is the capital of France?" in rendered

    def test_full_text_answer_format(self):
        assert self.formats.answer_format == "full_text"


# ---------------------------------------------------------------------------
# GenerateFormat
# ---------------------------------------------------------------------------


class TestGenerateFormat:
    """Test GenerateFormat defaults and Jinja generation."""

    def setup_method(self):
        self.formats = GenerateFormat()

    def test_defaults(self):
        assert self.formats.output_type == "generate_until"
        assert self.formats.choice_labels == "letters"
        assert self.formats.answer_format == "letters"
        assert self.formats.gen_prefix == "The best answer is"
        assert self.formats.scorer == "first_token"

    def test_has_instruction(self):
        assert self.formats.instruction is not None
        assert "choose the best answer" in self.formats.instruction

    def test_jinja_doc_to_text_renders(self):
        cfg = self.formats.to_jinja_config()
        rendered = render(cfg["doc_to_text"], MCQ_DOC)
        # Should start with the instruction
        assert "choose the best answer" in rendered
        # Should contain lettered choices
        assert "A." in rendered
        assert "D." in rendered
        # Should contain the question
        assert "What is the capital of France?" in rendered

    def test_jinja_doc_to_target_letter_format(self):
        """GenerateFormat with letters format converts index to letter."""
        cfg = self.formats.to_jinja_config()
        rendered = render(cfg["doc_to_target"], MCQ_DOC)
        assert rendered.strip() == "C"  # index 2 -> "C"

    def test_jinja_doc_to_target_text_answer(self):
        """GenerateFormat with letters format passes through text."""
        cfg = self.formats.to_jinja_config()
        rendered = render(cfg["doc_to_target"], TEXT_ANSWER_DOC)
        assert rendered.strip() == "Mercury"

    def test_to_task_config_has_gen_prefix(self):
        cfg = self.formats.to_task_config()
        assert cfg["gen_prefix"] == "The best answer is"

    def test_to_task_config_has_scorer(self):
        cfg = self.formats.to_task_config()
        assert cfg["scorer"] == "first_token"


# ---------------------------------------------------------------------------
# COTGenerateFormat
# ---------------------------------------------------------------------------


class TestCOTGenerateFormat:
    """Test COTGenerateFormat (chain-of-thought variant of GenerateFormat)."""

    def setup_method(self):
        self.formats = COTGenFormat()

    def test_inherits_generate(self):
        assert isinstance(self.formats, GenerateFormat)
        assert self.formats.output_type == "generate_until"

    def test_cot_instruction(self):
        assert "reason step by step" in self.formats.instruction

    def test_no_choice_labels(self):
        assert self.formats.choice_labels is None

    def test_no_gen_prefix(self):
        assert self.formats.gen_prefix is None

    def test_full_text_answer_format(self):
        assert self.formats.answer_format == "full_text"

    def test_question_prefix_is_problem(self):
        assert self.formats.question_prefix == "Problem: "

    def test_jinja_doc_to_text_no_choices(self):
        cfg = self.formats.to_jinja_config()
        rendered = render(cfg["doc_to_text"], GEN_DOC)
        # Should contain the instruction
        assert "reason step by step" in rendered
        # Should contain "Problem:" prefix
        assert "Problem:" in rendered
        # Should contain the question
        assert "What is 2 + 2?" in rendered
        # Should NOT contain choice labels (CoT has no choices)
        assert "A." not in rendered

    def test_jinja_doc_to_target_free_text(self):
        cfg = self.formats.to_jinja_config()
        rendered = render(cfg["doc_to_target"], GEN_DOC)
        assert rendered.strip() == "4"

    def test_to_task_config_no_gen_prefix(self):
        cfg = self.formats.to_task_config()
        assert "gen_prefix" not in cfg


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------


class TestOverrides:
    """Test that formats accept and apply field overrides."""

    def test_override_instruction(self):
        formats = MCQAFormat(instruction="Custom instruction here")
        cfg = formats.to_jinja_config()
        rendered = render(cfg["doc_to_text"], MCQ_DOC)
        assert "Custom instruction here" in rendered

    def test_override_question_prefix(self):
        formats = MCQAFormat(question_prefix="Q:")
        cfg = formats.to_jinja_config()
        rendered = render(cfg["doc_to_text"], MCQ_DOC)
        assert "Q:" in rendered
        assert "Question:" not in rendered

    def test_override_answer_prompt(self):
        formats = MCQAFormat(answer_prompt="The answer is:")
        cfg = formats.to_jinja_config()
        rendered = render(cfg["doc_to_text"], MCQ_DOC)
        assert rendered.rstrip().endswith("The answer is:")

    def test_override_choice_labels_numbers(self):
        formats = MCQAFormat(choice_labels="numbers")
        cfg = formats.to_jinja_config()
        rendered = render(cfg["doc_to_text"], MCQ_DOC)
        # Numbers are 1-based
        assert "1." in rendered
        assert "4." in rendered

    def test_override_choice_labels_custom_list(self):
        formats = MCQAFormat(choice_labels=["I", "II", "III", "IV"])
        cfg = formats.to_jinja_config()
        rendered = render(cfg["doc_to_text"], MCQ_DOC)
        assert "I." in rendered
        assert "IV." in rendered

    def test_override_via_get_dict(self):
        formats = FormatConfig.get({"type": "mcqa", "instruction": "Override via dict"})
        assert isinstance(formats, MCQAFormat)
        assert formats.instruction == "Override via dict"

    def test_override_target_delimiter(self):
        formats = MCQAFormat(target_delimiter="\n")
        cfg = formats.to_task_config()
        assert cfg["target_delimiter"] == "\n"


# ---------------------------------------------------------------------------
# doc_to_* field mapping
# ---------------------------------------------------------------------------


class TestFieldMapping:
    """Test that custom doc_to_* field names are correctly wired."""

    def test_custom_doc_to_text_field(self):
        formats = MCQAFormat()
        doc = {
            "prompt": "Custom question field",
            "choices": ["A", "B"],
            "answer": 0,
        }
        cfg = formats.to_jinja_config(doc_to_text="prompt")
        rendered = render(cfg["doc_to_text"], doc)
        assert "Custom question field" in rendered

    def test_custom_doc_to_choice_field(self):
        formats = MCQAFormat()
        doc = {
            "question": "Test",
            "options": ["Alpha", "Beta", "Gamma"],
            "answer": 1,
        }
        cfg = formats.to_jinja_config(doc_to_choice="options")
        rendered = render(cfg["doc_to_text"], doc)
        assert "Alpha" in rendered
        assert "Beta" in rendered

    def test_internal_jinja_list_doc_to_choice_field(self):
        formats = MCQAFormat()
        doc = {
            "question": "Test",
            "choice1": "Alpha",
            "choice2": "Beta",
            "choice3": "Gamma",
            "answer": 1,
        }
        cfg = formats.to_jinja_config(doc_to_choice="{{[choice1, choice2, choice3]}}")
        rendered = render(cfg["doc_to_text"], doc)
        assert "Alpha" in rendered
        assert "Beta" in rendered
        assert "Gamma" in rendered
        assert "A." in rendered
        assert "C." in rendered

    def test_custom_doc_to_target_field(self):
        formats = MCQAFormat()
        doc = {
            "question": "Test",
            "choices": ["X", "Y"],
            "label": 0,
        }
        cfg = formats.to_jinja_config(doc_to_target="label")
        rendered = render(cfg["doc_to_target"], doc)
        assert rendered.strip() == "0"

    def test_no_choices_field_generate(self):
        """When doc_to_choice is None on a generate formats, no choices section."""
        formats = GenerateFormat(choice_labels=None)
        cfg = formats.to_jinja_config(doc_to_choice=None)
        rendered = render(cfg["doc_to_text"], GEN_DOC)
        assert "A." not in rendered
        assert cfg["doc_to_choice"] is None

    def test_no_choices_field_mcq_raises(self):
        """multiple_choice output_type requires doc_to_choice."""
        formats = MCQAFormat()
        with pytest.raises(ValueError, match="no doc_to_choice was provided"):
            formats.to_jinja_config(doc_to_choice=None)

    def test_jinja_expression_in_field(self):
        """Field names that are already Jinja expressions should work."""
        formats = MCQAFormat()
        doc = {
            "doc": {"question": "Nested question"},
            "choices": ["a", "b"],
            "answer": 0,
        }
        cfg = formats.to_jinja_config(doc_to_text="{{doc['question']}}")
        rendered = render(cfg["doc_to_text"], doc)
        assert "Nested question" in rendered


# ---------------------------------------------------------------------------
# Target format variants
# ---------------------------------------------------------------------------


class TestTargetFormats:
    """Test different answer_format values in doc_to_target."""

    def test_letters_format_numeric_answer(self):
        formats = GenerateFormat(choice_labels="letters")
        cfg = formats.to_jinja_config()
        rendered = render(cfg["doc_to_target"], {"answer": 0, "choices": ["a"]})
        assert rendered.strip() == "A"

    def test_letters_format_text_answer(self):
        formats = GenerateFormat(choice_labels="letters")
        cfg = formats.to_jinja_config()
        rendered = render(cfg["doc_to_target"], {"answer": "B"})
        assert rendered.strip() == "B"

    def test_numbers_format_numeric_answer(self):
        formats = GenerateFormat(choice_labels="numbers")
        cfg = formats.to_jinja_config()
        rendered = render(cfg["doc_to_target"], {"answer": 0, "choices": ["a"]})
        # numbers is 1-based: index 0 -> "1"
        assert rendered.strip() == "1"

    def test_numbers_format_text_answer(self):
        formats = GenerateFormat(choice_labels="numbers")
        cfg = formats.to_jinja_config()
        rendered = render(cfg["doc_to_target"], {"answer": "42"})
        assert rendered.strip() == "42"

    def test_full_text_format_numeric_answer(self):
        formats = GenerateFormat(choice_labels=None)
        cfg = formats.to_jinja_config()
        rendered = render(cfg["doc_to_target"], MCQ_DOC)
        # index 2 -> choices[2] = "Paris"
        assert rendered.strip() == "Paris"

    def test_full_text_format_text_answer(self):
        formats = GenerateFormat(choice_labels=None)
        cfg = formats.to_jinja_config()
        rendered = render(cfg["doc_to_target"], {"answer": "freeform text"})
        assert rendered.strip() == "freeform text"

    def test_multiple_choice_target_numeric(self):
        """multiple_choice output_type returns index directly."""
        formats = MCQAFormat()
        cfg = formats.to_jinja_config()
        rendered = render(cfg["doc_to_target"], MCQ_DOC)
        assert rendered.strip() == "2"

    def test_multiple_choice_target_text_lookup(self):
        """multiple_choice output_type with text answer looks up index."""
        formats = MCQAFormat()
        cfg = formats.to_jinja_config()
        rendered = render(cfg["doc_to_target"], TEXT_ANSWER_DOC)
        # "Mercury" is at index 1
        assert rendered.strip() == "1"


# ---------------------------------------------------------------------------
# doc_to_choice Jinja
# ---------------------------------------------------------------------------


class TestDocToChoice:
    """Test _build_doc_to_choice_jinja output."""

    def test_letters_labels(self):
        formats = MCQAFormat(choice_labels="letters")
        cfg = formats.to_jinja_config()
        rendered = render(cfg["doc_to_choice"], MCQ_DOC)  # type:ignore[invalid-argument-type]
        # 4 choices -> ['A', 'B', 'C', 'D']
        assert "A" in rendered
        assert "D" in rendered

    def test_numbers_labels(self):
        formats = MCQAFormat(choice_labels="numbers")
        cfg = formats.to_jinja_config()
        rendered = render(cfg["doc_to_choice"], MCQ_DOC)  # type:ignore[invalid-argument-type]
        assert "1" in rendered
        assert "4" in rendered

    def test_custom_list_labels(self):
        formats = MCQAFormat(choice_labels=["X", "Y", "Z", "W"])
        cfg = formats.to_jinja_config()
        rendered = render(cfg["doc_to_choice"], MCQ_DOC)  # type:ignore[invalid-argument-type]
        assert "X" in rendered
        assert "W" in rendered

    def test_no_labels_returns_raw(self):
        formats = MCQAFormat(choice_labels=None)
        cfg = formats.to_jinja_config()
        rendered = render(cfg["doc_to_choice"], MCQ_DOC)  # type:ignore[invalid-argument-type]
        # Should output the raw choices list
        assert "Berlin" in rendered

    def test_no_choices_field_generate(self):
        """generate_until output_type can have doc_to_choice=None."""
        formats = GenerateFormat(choice_labels=None)
        cfg = formats.to_jinja_config(doc_to_choice=None)
        assert cfg["doc_to_choice"] is None


# ---------------------------------------------------------------------------
# Real-world doc_to_choice patterns from task YAMLs
# ---------------------------------------------------------------------------


class TestRealWorldDocToChoicePatterns:
    """Test common doc_to_choice patterns found across task YAML configs.

    Patterns sourced from lm_eval/tasks/:
    - Pattern 2: Hardcoded YAML list → Python list (e.g. pubmedqa)
    - Pattern 3: Jinja nested field  (e.g. arc_eu_easy: {{choices.text}})
    - Pattern 4: Jinja deep nested   (e.g. truthfulqa: {{mc1_targets.choices}})
    - Pattern 5: Jinja list construction (e.g. siqa: {{[answerA, answerB, answerC]}})
    - Pattern 6: Jinja map filter    (e.g. headqa: {{answers|map(attribute='atext')|list}})
    """

    def test_hardcoded_list(self):
        """Pattern 2: doc_to_choice: ["yes", "no", "maybe"] (YAML list → Python list)."""
        formats = MCQAFormat()
        doc = {"question": "Is this correct?", "choices": None, "answer": 0}
        cfg = formats.to_jinja_config(doc_to_choice=["yes", "no", "maybe"])
        # doc_to_text: should render the hardcoded choices
        text = render(cfg["doc_to_text"], doc)
        assert "yes" in text
        assert "no" in text
        assert "maybe" in text
        assert "A." in text and "C." in text
        # doc_to_target: numeric index
        target = render(cfg["doc_to_target"], doc)
        assert target.strip() == "0"
        # doc_to_choice: letter labels
        choice = render(cfg["doc_to_choice"], doc)  # type:ignore[arg-type]
        assert "A" in choice and "C" in choice

    def test_jinja_nested_field(self):
        """Pattern 3: doc_to_choice: "{{choices.text}}" (e.g. arc_eu_easy)."""
        formats = MCQAFormat()
        doc = {
            "question": "What orbits the Sun?",
            "choices": {"text": ["Earth", "Moon", "Sun", "Star"]},
            "answer": 0,
        }
        cfg = formats.to_jinja_config(doc_to_choice="{{choices.text}}")
        # doc_to_text
        text = render(cfg["doc_to_text"], doc)
        assert "Earth" in text and "Star" in text
        assert "A." in text and "D." in text
        # doc_to_target
        target = render(cfg["doc_to_target"], doc)
        assert target.strip() == "0"
        # doc_to_choice
        choice = render(cfg["doc_to_choice"], doc)  # type:ignore[arg-type]
        assert "A" in choice and "D" in choice

    def test_jinja_deep_nested(self):
        """Pattern 4: doc_to_choice: "{{mc1_targets.choices}}" (e.g. truthfulqa)."""
        formats = MCQAFormat()
        doc = {
            "question": "Is the earth flat?",
            "mc1_targets": {"choices": ["No", "Yes"]},
            "answer": 0,
        }
        cfg = formats.to_jinja_config(doc_to_choice="{{mc1_targets.choices}}")
        text = render(cfg["doc_to_text"], doc)
        assert "No" in text and "Yes" in text
        target = render(cfg["doc_to_target"], doc)
        assert target.strip() == "0"

    def test_jinja_list_construction(self):
        """Pattern 5: doc_to_choice: "{{[answerA, answerB, answerC]}}" (e.g. siqa).

        Tests all three templates, including doc_to_target and doc_to_choice
        which previously broke due to missing parenthesization.
        """
        formats = MCQAFormat()
        doc = {
            "question": "What happens next?",
            "answerA": "Go home",
            "answerB": "Stay put",
            "answerC": "Run away",
            "answer": 1,
        }
        cfg = formats.to_jinja_config(doc_to_choice="{{[answerA, answerB, answerC]}}")
        # doc_to_text
        text = render(cfg["doc_to_text"], doc)
        assert "Go home" in text and "Run away" in text
        assert "A." in text and "C." in text
        # doc_to_target (this was broken before the _field_ref fix)
        target = render(cfg["doc_to_target"], doc)
        assert target.strip() == "1"
        # doc_to_choice (this was broken before the _field_ref fix)
        choice = render(cfg["doc_to_choice"], doc)  # type:ignore[arg-type]
        assert "A" in choice and "C" in choice

    def test_jinja_list_construction_text_answer(self):
        """Pattern 5 with text answer: .index() lookup via parenthesized expression."""
        formats = MCQAFormat()
        doc = {
            "question": "Pick one",
            "answerA": "Alpha",
            "answerB": "Beta",
            "answer": "Beta",
        }
        cfg = formats.to_jinja_config(doc_to_choice="{{[answerA, answerB]}}")
        target = render(cfg["doc_to_target"], doc)
        assert target.strip() == "1"  # "Beta" is at index 1

    def test_jinja_map_filter(self):
        """Pattern 6: doc_to_choice: "{{answers|map(attribute='atext')|list}}" (e.g. headqa)."""
        formats = MCQAFormat()
        doc = {
            "question": "Which drug?",
            "answers": [
                {"atext": "Aspirin"},
                {"atext": "Ibuprofen"},
                {"atext": "Paracetamol"},
            ],
            "answer": 2,
        }
        cfg = formats.to_jinja_config(
            doc_to_choice="{{answers|map(attribute='atext')|list}}"
        )
        # doc_to_text
        text = render(cfg["doc_to_text"], doc)
        assert "Aspirin" in text and "Paracetamol" in text
        assert "A." in text and "C." in text
        # doc_to_target
        target = render(cfg["doc_to_target"], doc)
        assert target.strip() == "2"
        # doc_to_choice
        choice = render(cfg["doc_to_choice"], doc)  # type:ignore[arg-type]
        assert "A" in choice and "C" in choice

    def test_jinja_list_construction_generate_formats(self):
        """Pattern 5 with GenerateFormat: letter-format target from list construction."""
        formats = GenerateFormat()
        doc = {
            "question": "Best answer?",
            "answerA": "Option one",
            "answerB": "Option two",
            "answerC": "Option three",
            "answerD": "Option four",
            "answer": 2,
        }
        cfg = formats.to_jinja_config(
            doc_to_choice="{{[answerA, answerB, answerC, answerD]}}"
        )
        # doc_to_target: index 2 → "C" (letters format)
        target = render(cfg["doc_to_target"], doc)
        assert target.strip() == "C"

    def test_hardcoded_list_with_task_config(self):
        """Pattern 2 through to_task_config() — the full integration path."""
        formats = MCQAFormat()
        cfg = formats.to_task_config(doc_to_choice=["yes", "no"])
        assert cfg["output_type"] == "multiple_choice"
        doc = {"question": "Agree?", "answer": 1}
        text = render(cfg["doc_to_text"], doc)
        assert "yes" in text and "no" in text
        target = render(cfg["doc_to_target"], doc)
        assert target.strip() == "1"


# ---------------------------------------------------------------------------
# Real-world doc_to_text patterns from task YAMLs
# ---------------------------------------------------------------------------


class TestRealWorldDocToTextPatterns:
    """Test common doc_to_text patterns found across real task YAML configs.

    Patterns sourced from exploration of lm_eval/tasks/:
    - Plain field name           (e.g. openbookqa: question_stem)
    - Jinja with .strip()        (e.g. commonsense_qa: {{ question.strip() }})
    - Nested dict access         (e.g. arc: {{question.stem}})
    - Bracket dict access        (e.g. {{row['question']}})
    - Multi-field Jinja          (e.g. context + question combined)
    - Join expression            (e.g. xstorycloze: {{[s1,s2,s3]|join(' ')}})
    - String slicing             (e.g. lambada: {{text.split(' ')[:-1]|join(' ')}})
    """

    def test_plain_field_name(self):
        """Pattern: doc_to_text: question_stem (e.g. openbookqa)."""
        formats = MCQAFormat()
        doc = {"question_stem": "What causes rain?", "choices": ["A", "B"], "answer": 0}
        cfg = formats.to_jinja_config(doc_to_text="question_stem")
        rendered = render(cfg["doc_to_text"], doc)
        assert "What causes rain?" in rendered
        assert "Question:" in rendered

    def test_jinja_with_strip(self):
        """Pattern: doc_to_text: "{{ question.strip() }}" (e.g. commonsense_qa)."""
        formats = MCQAFormat()
        doc = {"question": "  Padded question  ", "choices": ["X", "Y"], "answer": 0}
        cfg = formats.to_jinja_config(doc_to_text="{{ question.strip() }}")
        rendered = render(cfg["doc_to_text"], doc)
        assert "Padded question" in rendered
        # Leading whitespace should be stripped by .strip()
        assert "  Padded" not in rendered

    def test_jinja_nested_dict_access(self):
        """Pattern: doc_to_text: "{{question.stem}}" (e.g. arc)."""
        formats = MCQAFormat()
        doc = {"question": {"stem": "Nested Q"}, "choices": ["A", "B"], "answer": 0}
        cfg = formats.to_jinja_config(doc_to_text="{{question.stem}}")
        rendered = render(cfg["doc_to_text"], doc)
        assert "Nested Q" in rendered

    def test_jinja_bracket_access(self):
        """Pattern: doc_to_text: "{{row['question']}}" (bracket dict access)."""
        formats = MCQAFormat()
        doc = {"row": {"question": "Bracket Q"}, "choices": ["A", "B"], "answer": 0}
        cfg = formats.to_jinja_config(doc_to_text="{{row['question']}}")
        rendered = render(cfg["doc_to_text"], doc)
        assert "Bracket Q" in rendered

    def test_multi_field_jinja(self):
        """Pattern: doc_to_text references multiple doc fields (context + question)."""
        formats = MCQAFormat()
        doc = {
            "context": "The sky is blue.",
            "question": "What color is the sky?",
            "choices": ["Red", "Blue"],
            "answer": 1,
        }
        cfg = formats.to_jinja_config(doc_to_text="{{context}}\n{{question}}")
        rendered = render(cfg["doc_to_text"], doc)
        assert "The sky is blue." in rendered
        assert "What color is the sky?" in rendered
        # Format structure should still wrap around it
        assert "Question:" in rendered
        assert "Answer:" in rendered

    def test_jinja_join_expression(self):
        """Pattern: doc_to_text: "{{[s1,s2,s3]|join(' ')}}" (e.g. xstorycloze)."""
        formats = GenerateFormat(choice_labels=None)
        doc = {
            "s1": "Once upon a time.",
            "s2": "There was a cat.",
            "s3": "The end.",
            "answer": "happy",
        }
        cfg = formats.to_jinja_config(
            doc_to_text="{{[s1, s2, s3]|join(' ')}}",
            doc_to_choice=None,
        )
        rendered = render(cfg["doc_to_text"], doc)
        assert "Once upon a time." in rendered
        assert "There was a cat." in rendered
        assert "The end." in rendered

    def test_jinja_string_manipulation(self):
        """Pattern: doc_to_text: "{{text.split(' ')[:-1]|join(' ')}}" (e.g. lambada)."""
        formats = GenerateFormat(choice_labels=None)
        doc = {"text": "The cat sat on the mat", "answer": "mat"}
        cfg = formats.to_jinja_config(
            doc_to_text="{{text.split(' ')[:-1]|join(' ')}}",
            doc_to_choice=None,
        )
        rendered = render(cfg["doc_to_text"], doc)
        # Should contain all words except the last
        assert "The cat sat on the" in rendered


# ---------------------------------------------------------------------------
# Integer doc_to_target (YAML `doc_to_target: 0`)
# ---------------------------------------------------------------------------


class TestIntegerDocToTarget:
    """Test that integer doc_to_target values work as constants."""

    def test_integer_target_zero_mcq(self):
        """doc_to_target=0 produces constant 0 (not swallowed by falsiness)."""
        formats = MCQAFormat()
        cfg = formats.to_jinja_config(doc_to_target=0)
        rendered = render(cfg["doc_to_target"], MCQ_DOC)
        assert rendered.strip() == "0"

    def test_integer_target_3_mcq(self):
        """doc_to_target=3 produces constant 3."""
        formats = MCQAFormat()
        cfg = formats.to_jinja_config(doc_to_target=3)
        rendered = render(cfg["doc_to_target"], MCQ_DOC)
        assert rendered.strip() == "3"

    def test_integer_target_generate_letters(self):
        """Integer target with letters format converts to letter."""
        formats = GenerateFormat()
        cfg = formats.to_jinja_config(doc_to_target=2)
        rendered = render(cfg["doc_to_target"], MCQ_DOC)
        # Constant 2 is a number → converted to letter "C"
        assert rendered.strip() == "C"

    def test_integer_target_generate_numbers(self):
        """Integer target with numbers format converts to 1-based."""
        formats = GenerateFormat(choice_labels="numbers")
        cfg = formats.to_jinja_config(doc_to_target=0)
        rendered = render(cfg["doc_to_target"], MCQ_DOC)
        # Constant 0 → 0 + 1 = 1
        assert rendered.strip() == "1"

    def test_integer_target_in_doc_to_text(self):
        """Integer doc_to_text is also coerced to Jinja literal."""
        formats = GenerateFormat(choice_labels=None)
        doc = {"question": "unused", "answer": "yes"}
        cfg = formats.to_jinja_config(doc_to_text=42, doc_to_choice=None)
        rendered = render(cfg["doc_to_text"], doc)
        assert "42" in rendered


# ---------------------------------------------------------------------------
# _escape_jinja contract
# ---------------------------------------------------------------------------


class TestEscapeJinja:
    """Document that _escape_jinja is intentionally a no-op identity function."""

    def test_escape_jinja_is_identity(self):
        assert FormatConfig._escape_jinja("Hello {{world}}") == "Hello {{world}}"
        assert (
            FormatConfig._escape_jinja("quotes 'and' \"double\"")
            == "quotes 'and' \"double\""
        )
        assert FormatConfig._escape_jinja("") == ""


# ---------------------------------------------------------------------------
# TaskConfig._resolve_formats() integration
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> TaskConfig:
    """Build a minimal TaskConfig for testing formats resolution.

    Includes doc_to_choice="choices" by default because MCQ formats
    require a choices field for template generation.
    """
    defaults: dict = {
        "task": "test_task",
        "dataset_path": "dummy",
        "test_split": "test",
        "doc_to_choice": "choices",
    }
    defaults.update(overrides)
    return TaskConfig(**defaults)


class TestTaskConfigResolveFormat:
    """Test TaskConfig.__post_init__ -> _resolve_formats() integration.

    Verifies that the @suffix parsing, formats field resolution, and
    override application in TaskConfig work correctly end-to-end.
    """

    def test_at_suffix_parses_formats_selection(self):
        """task='t@mcqa' sets _formats_selection and formats."""
        cfg = _make_config(task="t@mcqa")
        assert cfg._formats_selection == "mcqa"
        assert cfg.formats == "mcqa"

    def test_at_suffix_applies_mcq_formats(self):
        """task='t@mcqa' applies MCQFormat: output_type, doc_to_text, etc."""
        cfg = _make_config(task="t@mcqa")
        assert cfg.output_type == "multiple_choice"
        assert "Question:" in cfg.doc_to_text
        assert cfg.doc_to_target is not None
        assert cfg.doc_to_choice is not None
        assert cfg.target_delimiter == " "
        assert cfg.fewshot_delimiter == "\n\n"

    def test_at_suffix_cloze_formats(self):
        """task='t@cloze' applies ClozeFormat (MCQ variant, no choice labels)."""
        cfg = _make_config(task="t@cloze")
        assert cfg.output_type == "multiple_choice"
        # ClozeFormat has choice_labels=None, so the for-loop in the template
        # uses empty string labels instead of A/B/C
        assert cfg.doc_to_text is not None

    def test_at_suffix_generate_formats(self):
        """task='t@generate' applies GenerateFormat."""
        cfg = _make_config(task="t@generate")
        assert cfg.output_type == "generate_until"
        assert "choose the best answer" in cfg.doc_to_text

    def test_explicit_formats_field_no_at_suffix(self):
        """formats='mcqa' with no @ in task name applies formats."""
        cfg = _make_config(formats="mcqa")
        assert cfg.output_type == "multiple_choice"
        assert cfg._formats_selection is None
        assert cfg.formats == "mcqa"

    def test_multi_formats_with_at_suffix_selection(self):
        """Multi-formats dict + @suffix selects the right variant."""
        cfg = _make_config(
            task="t@generate",
            formats={
                "mcqa": {"instruction": "MCQ here"},
                "generate": {"instruction": "Gen here"},
            },
        )
        assert cfg.output_type == "generate_until"
        assert "Gen here" in cfg.doc_to_text

    def test_multi_formats_default_first_key(self):
        """Multi-formats dict with no @suffix defaults to first key."""
        cfg = _make_config(
            formats={"mcqa": None, "generate": None},
        )
        assert cfg.output_type == "multiple_choice"

    def test_doc_to_text_default_fallback(self):
        """When doc_to_text is None, formats uses 'question' as default field."""
        cfg = _make_config(task="t@mcqa")
        # doc_to_text was None -> formats used "question" as field name
        assert "question" in cfg.doc_to_text

    def test_doc_to_target_default_fallback(self):
        """When doc_to_target is None, formats uses 'answer' as default field."""
        cfg = _make_config(task="t@mcqa")
        # The target template references 'answer'
        assert "answer" in cfg.doc_to_target

    def test_explicit_doc_to_text_overrides_default(self):
        """Explicit doc_to_text='prompt' is used instead of default 'question'."""
        cfg = _make_config(task="t@mcqa", doc_to_text="prompt_field")
        assert "prompt_field" in cfg.doc_to_text
        # Render the template to confirm it references the custom field
        rendered = render(
            cfg.doc_to_text, {"prompt_field": "Hello", "choices": ["a", "b"]}
        )
        assert "Hello" in rendered

    def test_explicit_doc_to_choice_passed_through(self):
        """Explicit doc_to_choice='options' is used in formats template."""
        cfg = _make_config(task="t@mcqa", doc_to_choice="options")
        assert "options" in cfg.doc_to_text

    def test_formats_selection_runtime_override(self):
        """_formats_selection='cloze' without @ in task name applies ClozeFormat."""
        cfg = _make_config(_formats_selection="cloze")
        assert cfg.output_type == "multiple_choice"
        assert cfg.formats == "cloze"
        assert cfg._formats_selection == "cloze"

    def test_formats_selection_priority_over_suffix(self):
        """Runtime _formats_selection takes priority over @suffix in task name."""
        cfg = _make_config(task="t@generate", _formats_selection="mcqa")
        # _formats_selection="mcqa" was already set, so @ parsing is skipped
        assert cfg._formats_selection == "mcqa"
        assert cfg.output_type == "multiple_choice"

    def test_unknown_formats_in_at_suffix_raises(self):
        """Unknown formats name in @suffix raises ValueError."""
        with pytest.raises(ValueError, match="Unknown format"):
            _make_config(task="t@nonexistent_formats_xyz")


# ---------------------------------------------------------------------------
# TaskConfig formats scorer integration
# ---------------------------------------------------------------------------


class TestTaskConfigFormatScorer:
    """Test that formats scorer fields survive TaskConfig normalization."""

    def test_mcq_scorer_is_none(self):
        """MCQFormat has scorer=None — stays None after normalization."""
        cfg = _make_config(task="t@mcqa")
        assert cfg.scorer is None

    def test_generate_scorer_normalized_to_dict(self):
        """GenerateFormat scorer='first_token' is normalized to dict form."""
        cfg = _make_config(task="t@generate")
        # _normalize_scoring_config converts str -> {"type": str}
        assert cfg.scorer == {"type": "first_token"}
