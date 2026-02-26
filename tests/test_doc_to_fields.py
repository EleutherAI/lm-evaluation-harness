"""First-pass tests for doc_to_* field processing.

Tests the core parsing functions (process_field, _coerce_list, _coerce_target)
and the Task.doc_to_text / doc_to_target / doc_to_choice methods to make sure
various config inputs (plain strings, digit strings, Jinja templates, lists,
callables, etc.) are resolved correctly.
"""

from __future__ import annotations

from lm_eval.api.task import (
    LoglikelihoodTask,
    MultipleChoiceTask,
    Task,
)
from lm_eval.config.task import TaskConfig
from lm_eval.config.utils import (
    _coerce_list,
    _coerce_target,
    _resolve_target_index,
    process_field,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DOC = {"question": "What colour is the sky?", "answer": "blue", "label": "1", "idx": 3}


def _make_task(**overrides) -> Task:
    """Build a minimal Task (generate_until) for testing doc_to_* methods."""
    defaults = {
        "task": "test_task",
        "output_type": "generate_until",
        "dataset_path": "dummy",
        "test_split": "test",
    }
    defaults.update(overrides)
    return Task(TaskConfig(**defaults))  # type:ignore[invalid-argument-type]


def _make_mc_task(**overrides) -> MultipleChoiceTask:
    """Build a minimal MultipleChoiceTask."""
    defaults = {
        "task": "test_mc",
        "output_type": "multiple_choice",
        "dataset_path": "dummy",
        "test_split": "test",
    }
    defaults.update(overrides)
    return MultipleChoiceTask(TaskConfig(**defaults))  # type:ignore[invalid-argument-type]


def _make_ll_task(**overrides) -> LoglikelihoodTask:
    """Build a minimal LoglikelihoodTask."""
    defaults = {
        "task": "test_ll",
        "output_type": "loglikelihood",
        "dataset_path": "dummy",
        "test_split": "test",
    }
    defaults.update(overrides)
    return LoglikelihoodTask(TaskConfig(**defaults))  # type:ignore[invalid-argument-type]


# ===================================================================
# process_field
# ===================================================================


class TestProcessField:
    """Unit tests for the core dispatch in process_field()."""

    def test_none(self):
        assert process_field(DOC, None) is None

    def test_int_passthrough(self):
        assert process_field(DOC, 3) == 3
        assert process_field(DOC, 0) == 0

    def test_callable(self):
        assert process_field(DOC, lambda d: d["answer"].upper()) == "BLUE"

    # -- string: key lookup vs template --

    def test_string_key_lookup(self):
        """A plain string that matches a doc key returns the value directly."""
        assert process_field(DOC, "answer") == "blue"

    def test_string_key_returns_int(self):
        """Key lookup returns whatever the doc value is, even an int."""
        assert process_field({"count": 42}, "count") == 42

    def test_jinja_simple_var(self):
        assert process_field(DOC, "{{answer}}") == "blue"

    def test_jinja_with_surrounding_text(self):
        result = process_field(DOC, "Q: {{question}}\nA:")
        assert result == "Q: What colour is the sky?\nA:"

    def test_jinja_expression(self):
        """Jinja can evaluate expressions."""
        assert process_field(DOC, "{{idx + 1}}") == "4"

    def test_plain_string_no_key_no_template(self):
        """A string that isn't a key and has no {{ }} is still passed through
        apply_template (which just returns it unchanged).
        """
        assert process_field(DOC, "hello world") == "hello world"

    def test_digit_string_not_a_key(self):
        """'43' is not a doc key → rendered as a Jinja template → returns '43'."""
        assert process_field(DOC, "43") == "43"

    # -- list --

    def test_list_of_templates(self):
        result = process_field(DOC, ["{{answer}}", "{{question}}"])
        assert result == ["blue", "What colour is the sky?"]

    def test_list_of_plain_strings(self):
        result = process_field(DOC, ["yes", "no"])
        assert result == ["yes", "no"]

    def test_list_with_ints(self):
        """Non-string items in a list are kept as-is."""
        result = process_field(DOC, [1, "{{answer}}", 2])
        assert result == [1, "blue", 2]

    def test_empty_list(self):
        assert process_field(DOC, []) == []


# ===================================================================
# _coerce_list
# ===================================================================


class TestCoerceList:
    def test_actual_list_passthrough(self):
        assert _coerce_list(["a", "b"]) == ["a", "b"]

    def test_string_list_literal(self):
        """A string that looks like a Python list gets parsed."""
        assert _coerce_list("['yes', 'no']") == ["yes", "no"]

    def test_string_list_with_ints(self):
        assert _coerce_list("[1, 2, 3]") == [1, 2, 3]

    def test_plain_string(self):
        """A non-list string is returned as-is."""
        assert _coerce_list("hello") == "hello"

    def test_none(self):
        assert _coerce_list(None) is None

    def test_int_passthrough(self):
        assert _coerce_list(5) == 5

    def test_malformed_list_string(self):
        """Malformed list literal → returned as-is (no crash)."""
        assert _coerce_list("[not, valid python]") == "[not, valid python]"


# ===================================================================
# _coerce_target
# ===================================================================


class TestCoerceTarget:
    def test_digit_string_to_int(self):
        assert _coerce_target("43") == 43
        assert _coerce_target("0") == 0

    def test_non_digit_string_unchanged(self):
        assert _coerce_target("blue") == "blue"

    def test_int_passthrough(self):
        assert _coerce_target(3) == 3

    def test_none_passthrough(self):
        assert _coerce_target(None) is None

    def test_list_passthrough(self):
        assert _coerce_target(["a", "b"]) == ["a", "b"]

    def test_parse_list_false_ignores_list_string(self):
        """Without parse_list, a list-looking string stays a string."""
        assert _coerce_target("['a', 'b']", parse_list=False) == "['a', 'b']"

    def test_parse_list_true_parses_list_string(self):
        assert _coerce_target("['a', 'b']", parse_list=True) == ["a", "b"]

    def test_digit_takes_precedence_over_parse_list(self):
        """A digit string is coerced to int even when parse_list=True."""
        assert _coerce_target("7", parse_list=True) == 7


# ===================================================================
# Task.doc_to_text
# ===================================================================


class TestTaskDocToText:
    def test_jinja_template(self):
        t = _make_task(doc_to_text="Q: {{question}}")
        assert t.doc_to_text(DOC) == "Q: What colour is the sky?"

    def test_key_lookup(self):
        t = _make_task(doc_to_text="answer")
        assert t.doc_to_text(DOC) == "blue"

    def test_none_config(self):
        t = _make_task(doc_to_text=None)
        assert t.doc_to_text(DOC) is None

    def test_override_parameter(self):
        """The override parameter takes precedence over config."""
        t = _make_task(doc_to_text="answer")
        assert (
            t.doc_to_text(DOC, doc_to_text="{{question}}") == "What colour is the sky?"
        )

    def test_callable_config(self):
        t = _make_task(doc_to_text=lambda d: d["answer"].upper())
        assert t.doc_to_text(DOC) == "BLUE"

    def test_plain_literal_string(self):
        """A string that isn't a key or template just passes through."""
        t = _make_task(doc_to_text="Always answer yes.")
        assert t.doc_to_text(DOC) == "Always answer yes."


# ===================================================================
# Task.doc_to_target
# ===================================================================


class TestTaskDocToTarget:
    def test_digit_string_coerced_to_int(self):
        """Config value '1' (looks like a digit) → Jinja renders '1' → coerced to int 1."""
        doc = {"label": "1"}
        t = _make_task(doc_to_target="{{label}}")
        assert t.doc_to_target(doc) == 1

    def test_digit_string_literal(self):
        """Config is literally '43' (not a doc key) → rendered as '43' → coerced to 43."""
        t = _make_task(doc_to_target="43")
        assert t.doc_to_target(DOC) == 43

    def test_key_lookup_digit(self):
        """Config 'label' is a doc key whose value is '1' (str) → coerced to 1."""
        t = _make_task(doc_to_target="label")
        # DOC["label"] = "1"  →  _coerce_target("1") → 1
        assert t.doc_to_target(DOC) == 1

    def test_key_lookup_string(self):
        t = _make_task(doc_to_target="answer")
        assert t.doc_to_target(DOC) == "blue"

    def test_callable(self):
        t = _make_task(doc_to_target=lambda d: d["idx"])
        assert t.doc_to_target(DOC) == 3

    def test_int_config(self):
        """An int config is returned directly (no coercion needed)."""
        t = _make_task(doc_to_target="{{idx}}")
        # Jinja renders to "3" (str) → _coerce_target → 3
        assert t.doc_to_target(DOC) == 3

    def test_none_config(self):
        t = _make_task(doc_to_target=None)
        assert t.doc_to_target(DOC) is None

    def test_multiple_targets_parses_list(self):
        """With multiple_targets=True, a rendered list literal gets parsed."""
        doc = {"targets": "['cat', 'dog']"}
        t = _make_task(doc_to_target="{{targets}}", multiple_targets=True)
        assert t.doc_to_target(doc) == ["cat", "dog"]


# ===================================================================
# Task.doc_to_choice
# ===================================================================


class TestTaskDocToChoice:
    def test_static_list(self):
        t = _make_task(doc_to_choice=["yes", "no"])
        assert t.doc_to_choice(DOC) == ["yes", "no"]

    def test_template_list(self):
        """A list of Jinja templates, each rendered against the doc."""
        t = _make_task(doc_to_choice=["{{answer}}", "{{question}}"])
        assert t.doc_to_choice(DOC) == ["blue", "What colour is the sky?"]

    def test_key_lookup_to_list(self):
        """Config string is a doc key whose value is a list → returned as list."""
        doc = {"choices": ["A", "B", "C"]}
        t = _make_task(doc_to_choice="choices")
        assert t.doc_to_choice(doc) == ["A", "B", "C"]

    def test_template_renders_list_literal(self):
        """Jinja renders a list-looking string → _coerce_list parses it."""
        doc = {"opts": "['red', 'green', 'blue']"}
        t = _make_task(doc_to_choice="{{opts}}")
        assert t.doc_to_choice(doc) == ["red", "green", "blue"]

    def test_none_config(self):
        t = _make_task(doc_to_choice=None)
        assert t.doc_to_choice(DOC) is None

    def test_callable(self):
        t = _make_task(doc_to_choice=lambda d: ["x", "y"])
        assert t.doc_to_choice(DOC) == ["x", "y"]

    def test_non_list_returns_none(self):
        """If the result isn't a list (and isn't None), it should return None."""
        t = _make_task(doc_to_choice="answer")  # resolves to "blue" (str)
        assert t.doc_to_choice(DOC) is None


# ===================================================================
# _resolve_target_index
# ===================================================================


class TestResolveTargetIndex:
    """Unit tests for _resolve_target_index (used by MultipleChoiceTask)."""

    def test_int_in_range(self):
        assert _resolve_target_index(0, ["a", "b", "c"], {}) == 0
        assert _resolve_target_index(2, ["a", "b", "c"], {}) == 2

    def test_int_out_of_range(self):
        assert _resolve_target_index(5, ["a", "b"], {}) is None

    def test_string_match(self):
        assert _resolve_target_index("b", ["a", "b", "c"], {}) == 1

    def test_string_no_match(self):
        assert _resolve_target_index("z", ["a", "b"], {}) is None

    def test_float_truncated(self):
        assert _resolve_target_index(1.9, ["a", "b", "c"], {}) == 1


# ===================================================================
# MultipleChoiceTask.doc_to_target  (resolves to choice index)
# ===================================================================


MC_DOC = {
    "question": "Capital of France?",
    "choices": ["Berlin", "Paris", "Rome"],
    "answer": "1",
}


class TestMCDocToTarget:
    """MultipleChoiceTask.doc_to_target resolves the base target to a choice index."""

    def test_int_target_is_index(self):
        """Target '1' → base coerces to int 1 → MC resolves index 1 (Paris)."""
        t = _make_mc_task(doc_to_target="{{answer}}", doc_to_choice="choices")
        assert t.doc_to_target(MC_DOC) == 1

    def test_string_target_matched_in_choices(self):
        """Target is a string that matches a choice → returns its index."""
        doc = {"text": "Paris", "choices": ["Berlin", "Paris", "Rome"]}
        t = _make_mc_task(doc_to_target="text", doc_to_choice="choices")
        assert t.doc_to_target(doc) == 1

    def test_target_string_not_in_choices(self):
        """Target string not in choices → returns None."""
        doc = {"text": "London", "choices": ["Berlin", "Paris"]}
        t = _make_mc_task(doc_to_target="text", doc_to_choice="choices")
        assert t.doc_to_target(doc) is None

    def test_target_index_out_of_range(self):
        """Target int beyond choice length → returns None."""
        doc = {"idx": "5", "choices": ["a", "b"]}
        t = _make_mc_task(doc_to_target="idx", doc_to_choice="choices")
        assert t.doc_to_target(doc) is None

    def test_jinja_computed_index(self):
        """Jinja expression computing an index, e.g. arc_easy style."""
        doc = {
            "choices": {"text": ["Add", "Sub", "Mul"], "label": ["A", "B", "C"]},
            "answerKey": "B",
        }
        t = _make_mc_task(
            doc_to_target="{{choices.label.index(answerKey)}}",
            doc_to_choice=["Add", "Sub", "Mul"],
        )
        # Jinja renders "1" → coerced to int 1 → index 1
        assert t.doc_to_target(doc) == 1

    def test_digit_string_literal_43(self):
        """Config is literally '43' → rendered as '43' → coerced to 43 → out of range."""
        doc = {"choices": ["a", "b", "c"]}
        t = _make_mc_task(doc_to_target="43", doc_to_choice="choices")
        assert t.doc_to_target(doc) is None

    def test_zero_index(self):
        doc = {"label": "0", "choices": ["no", "yes"]}
        t = _make_mc_task(doc_to_target="label", doc_to_choice="choices")
        assert t.doc_to_target(doc) == 0


# ===================================================================
# MultipleChoiceTask.doc_to_choice
# ===================================================================


class TestMCDocToChoice:
    def test_static_list(self):
        t = _make_mc_task(doc_to_choice=["no", "yes"])
        assert t.doc_to_choice(MC_DOC) == ["no", "yes"]

    def test_key_lookup(self):
        t = _make_mc_task(doc_to_choice="choices")
        assert t.doc_to_choice(MC_DOC) == ["Berlin", "Paris", "Rome"]

    def test_template_list(self):
        doc = {"a": "cat", "b": "dog"}
        t = _make_mc_task(doc_to_choice=["{{a}}", "{{b}}"])
        assert t.doc_to_choice(doc) == ["cat", "dog"]

    def test_template_renders_list_literal(self):
        doc = {"opts": "['red', 'blue']"}
        t = _make_mc_task(doc_to_choice="{{opts}}")
        assert t.doc_to_choice(doc) == ["red", "blue"]

    def test_digit_like_literals(self):
        doc = {"opts": "['1', '2']"}
        t = _make_mc_task(doc_to_choice="{{opts}}")
        assert t.doc_to_choice(doc) == ["1", "2"]

    def test_callable(self):
        t = _make_mc_task(doc_to_choice=lambda d: d["choices"])
        assert t.doc_to_choice(MC_DOC) == ["Berlin", "Paris", "Rome"]


# ===================================================================
# MultipleChoiceTask.doc_to_text
# ===================================================================


class TestMCDocToText:
    def test_jinja_template(self):
        t = _make_mc_task(doc_to_text="Q: {{question}}")
        assert t.doc_to_text(MC_DOC) == "Q: Capital of France?"

    def test_key_lookup(self):
        t = _make_mc_task(doc_to_text="question")
        assert t.doc_to_text(MC_DOC) == "Capital of France?"

    def test_multiple_inputs_coerces_list(self):
        """With multiple_inputs=True, doc_to_text coerces a list-literal string."""
        doc = {"contexts": "['ctx1', 'ctx2']"}
        t = _make_mc_task(
            doc_to_text="{{contexts}}",
            doc_to_choice=["only_choice"],
            multiple_inputs=True,
        )
        assert t.doc_to_text(doc) == ["ctx1", "ctx2"]

    def test_multiple_inputs_list_config(self):
        """With multiple_inputs=True and a list config, templates are rendered."""
        doc = {"a": "first", "b": "second"}
        t = _make_mc_task(
            doc_to_text=["{{a}}", "{{b}}"],
            doc_to_choice=["only_choice"],
            multiple_inputs=True,
        )
        assert t.doc_to_text(doc) == ["first", "second"]


# ===================================================================
# LoglikelihoodTask.doc_to_*
# ===================================================================


class TestLLTaskDocTo:
    """LoglikelihoodTask uses the base Task doc_to_* methods (no overrides)."""

    def test_doc_to_text(self):
        t = _make_ll_task(doc_to_text="Q: {{question}}")
        assert t.doc_to_text(DOC) == "Q: What colour is the sky?"

    def test_doc_to_target_string(self):
        """LL tasks typically have a string target (the continuation)."""
        t = _make_ll_task(doc_to_target="answer")
        assert t.doc_to_target(DOC) == "blue"

    def test_doc_to_target_template(self):
        t = _make_ll_task(doc_to_target="{{answer}}")
        assert t.doc_to_target(DOC) == "blue"

    def test_doc_to_target_digit_is_coerced_so_invalid(self):
        """Even in LL tasks, digit strings get coerced to int by base Task."""
        t = _make_ll_task(doc_to_target="label")
        assert t.doc_to_target(DOC) is None
