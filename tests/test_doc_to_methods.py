"""Tests for Task doc_to_* methods with Jinja/YAML parsing.

This test suite documents and validates all expected YAML input types for the doc_to_* methods:

doc_to_text - Transforms a document into the input text for the model:
  - String field name: References a field directly from the document
    YAML: doc_to_text: "question"

  - Jinja2 template: Renders a template with document fields
    YAML: doc_to_text: "Question: {{question}}\nContext: {{context}}"

  - Integer: Returns a constant integer value
    YAML: doc_to_text: 0

  - Python function: Applies a callable function (via !function directive)
    YAML: doc_to_text: !function utils.my_custom_function

doc_to_target - Transforms a document into the expected target/answer:
  - String field name: References a field directly from the document
    YAML: doc_to_target: "answer"

  - Jinja2 template: Renders a template, can return string or int for multiple choice
    YAML: doc_to_target: "{{answers[correct_idx]}}"
    YAML: doc_to_target: "{{label}}" # "0", "1", etc. converted to int if doc_to_choice exists

  - Integer: Returns a constant integer value (typically for multiple choice)
    YAML: doc_to_target: 0

  - List of templates: Returns multiple targets: list[str]
    YAML: doc_to_target: ["{{answer1}}", "{{answer2}}"]

  - Python function: Applies a callable function
    YAML: doc_to_target: !function utils.extract_answer

doc_to_choice - Defines the list of choices for multiple choice tasks:
  - String field name: References a list field from the document
    YAML: doc_to_choice: "options"

  - Jinja2 template returning list: Template that evaluates to a list
    YAML: doc_to_choice: "{{choices}}" # Must render to "['A', 'B', 'C']" format
    YAML: doc_to_choice: "{{[correct, wrong]}}" # Creates list literal from fields
    YAML: doc_to_choice: "{{options if options else default_options}}"

  - List of templates: Each template becomes a choice
    YAML: doc_to_choice: ["{{choice_a}}", "{{choice_b}}", "{{choice_c}}"]

  - Dictionary: Values become the choices (keys are ignored)
    YAML: doc_to_choice:
      A: "First option"
      B: "Second option"
      C: "Third option"

  - Python function: Returns a list of choices
    YAML: doc_to_choice: !function utils.generate_choices

Special Jinja2 features supported:
  - Filters: {{text|upper}}, {{text|lower}}, {{text|regex_replace('pattern', 'replacement')}}
  - Conditionals: {{field1 if condition else field2}}
  - List operations: {{', '.join(items)}}
  - Nested field access: {{metadata.answer}}, {{choices[0]}}
  - Math operations: {{score * 100}}
  - String concatenation: {{first + ' ' + last}}
"""

from unittest.mock import Mock, patch

import pytest

from lm_eval.api.task import Task


class TestDocToTextMethod:
    """Test suite for doc_to_text method."""

    def test_doc_to_text_with_string_field(self):
        """Test doc_to_text when config points to a field name."""
        task = Mock(spec=Task)
        task.multiple_inputs = False
        task.features = ["text", "answer", "choices", "label"]
        task.config = Mock()
        task.config.doc_to_text = "text"

        doc = {"text": "This is a test question", "answer": "A"}

        result = Task.doc_to_text(task, doc)
        assert result == "This is a test question"

    def test_doc_to_text_with_jinja_template(self):
        """Test doc_to_text with Jinja template."""
        task = Mock(spec=Task)
        task.multiple_inputs = False
        task.features = ["text", "answer"]
        task.config = Mock()
        task.config.doc_to_text = "Question: {{text}}"

        doc = {"text": "What is 2+2?", "answer": "4"}

        result = Task.doc_to_text(task, doc)
        assert result == "Question: What is 2+2?"

    def test_doc_to_text_with_complex_jinja(self):
        """Test doc_to_text with complex Jinja expressions."""
        task = Mock(spec=Task)
        task.multiple_inputs = False
        task.features = ["text", "answer"]
        task.config = Mock()
        task.config.doc_to_text = "{{text|upper}} - {{answer|lower}}"

        doc = {"text": "Test", "answer": "ANSWER"}

        result = Task.doc_to_text(task, doc)
        assert result == "TEST - answer"

    def test_doc_to_text_with_list(self):
        """Test doc_to_text when config is an integer."""
        task = Mock(spec=Task)
        task.multiple_inputs = False
        task.config = Mock()
        task.config.doc_to_text = ["{{choice1}}", "{{choice2}}"]

        doc = {"choice1": "1", "choice2": "2"}

        result = Task.doc_to_text(task, doc)
        assert result == ["1", "2"]

    def test_doc_to_text_with_callable(self):
        """Test doc_to_text with a callable function."""

        def custom_text_func(doc):
            return f"Custom: {doc['text']}"

        task = Mock(spec=Task)
        task.multiple_inputs = False
        task.config = Mock()
        task.config.doc_to_text = custom_text_func

        doc = {"text": "test"}

        result = Task.doc_to_text(task, doc)
        assert result == "Custom: test"

    def test_doc_to_text_with_regex_filter(self):
        """Test doc_to_text with Jinja regex_replace filter."""
        task = Mock(spec=Task)
        task.multiple_inputs = False
        task.features = ["text"]
        task.config = Mock()
        task.config.doc_to_text = "{{text|regex_replace('\\d+', 'X')}}"

        doc = {"text": "There are 123 apples and 456 oranges"}

        result = Task.doc_to_text(task, doc)
        assert result == "There are X apples and X oranges"

    def test_doc_to_text_with_list_comprehension(self):
        """Test doc_to_text with Jinja list comprehension."""
        task = Mock(spec=Task)
        task.multiple_inputs = False
        task.features = []
        task.config = Mock()
        task.config.doc_to_text = "Options: {{ ', '.join(choices) }}"

        doc = {"choices": ["red", "green", "blue"]}

        result = Task.doc_to_text(task, doc)
        assert result == "Options: red, green, blue"

    def test_override_doc_to_text(self):
        """Test overriding doc_to_text with parameter."""
        task = Mock(spec=Task)
        task.multiple_inputs = False
        task.features = []
        task.config = Mock()
        task.config.doc_to_text = "default"

        doc = {"text": "test"}

        result = Task.doc_to_text(task, doc, doc_to_text="override")
        assert result == "override"

    def test_doc_to_text_type_error(self):
        """Test doc_to_text raises TypeError for invalid type."""
        task = Mock(spec=Task)
        task.multiple_inputs = False
        task.config = Mock()
        task.config.doc_to_text = {"invalid": "type"}

        doc = {"text": "test"}

        with pytest.raises(TypeError):
            Task.doc_to_text(task, doc)

    def test_doc_to_text_with_missing_field(self):
        """Test doc_to_text with missing field in template."""
        task = Mock(spec=Task)
        task.multiple_inputs = False
        task.features = []
        task.config = Mock()
        task.config.doc_to_text = "{{missing_field}}"

        doc = {"text": "test"}

        from jinja2 import UndefinedError

        with pytest.raises(UndefinedError):
            Task.doc_to_text(task, doc)


class TestDocToTargetMethod:
    """Test suite for doc_to_target method."""

    def test_doc_to_target_with_field(self):
        """Test doc_to_target when config points to a field name."""
        task = Mock(spec=Task)
        task.features = ["text", "answer"]
        task.config = Mock()
        task.config.doc_to_target = "answer"
        task._config = task.config

        doc = {"text": "question", "answer": "correct answer"}

        result = Task.doc_to_target(task, doc)
        assert result == "correct answer"

    def test_doc_to_target_with_jinja_template(self):
        """Test doc_to_target with Jinja template."""
        task = Mock(spec=Task)
        task.features = []
        task.config = Mock()
        task.config.doc_to_target = "{{answer}}"
        task.config.doc_to_choice = None
        task._config = task.config

        doc = {"answer": "test_answer"}

        result = Task.doc_to_target(task, doc)
        assert result == "test_answer"

    def test_doc_to_target_with_jinja_index(self):
        """Test doc_to_target with Jinja template returning numeric string."""
        task = Mock(spec=Task)
        task.features = []
        task.config = Mock()
        task.config.doc_to_target = "{{label}}"
        task.config.doc_to_choice = ["A", "B", "C"]
        task._config = task.config

        doc = {"label": "1"}

        result = Task.doc_to_target(task, doc)
        assert result == 1  # Should be converted to int

    def test_doc_to_target_with_int(self):
        """Test doc_to_target when config is an integer."""
        task = Mock(spec=Task)
        task.config = Mock()
        task.config.doc_to_target = 0
        task._config = task.config

        doc = {"answer": "test"}

        result = Task.doc_to_target(task, doc)
        assert result == 0

    def test_doc_to_target_with_list(self):
        """Test doc_to_target with list of templates."""
        task = Mock(spec=Task)
        task.features = []
        task.config = Mock()
        task.config.doc_to_target = ["{{answer}}", "{{text}}"]
        task._config = task.config

        doc = {"answer": "A", "text": "question"}

        result = Task.doc_to_target(task, doc)
        assert result == ["A", "question"]

    def test_doc_to_target_with_int_list(self):
        """Test doc_to_target with list of templates."""
        task = Mock(spec=Task)
        task.features = []
        task.multiple_targets = True
        task.config = Mock()
        task.config.doc_to_target = "{{answer}}"
        task._config = task.config

        doc = {"answer": [1, 2, 3, 4]}

        result = Task.doc_to_target(task, doc)
        assert result == [1, 2, 3, 4]

    def test_doc_to_target_with_callable(self):
        """Test doc_to_target with a callable function."""

        def custom_target_func(doc):
            return doc["label"] * 2

        task = Mock(spec=Task)
        task.config = Mock()
        task.config.doc_to_target = custom_target_func
        task._config = task.config

        doc = {"label": 3}

        result = Task.doc_to_target(task, doc)
        assert result == 6

    def test_doc_to_target_with_nested_fields(self):
        """Test doc_to_target with nested field access."""
        task = Mock(spec=Task)
        task.features = []
        task.config = Mock()
        task.config.doc_to_target = "{{meta.answer}}"
        task.config.doc_to_choice = None
        task._config = task.config

        doc = {"meta": {"answer": "nested_value"}}

        result = Task.doc_to_target(task, doc)
        assert result == "nested_value"

    def test_doc_to_target_multiple_targets(self):
        """Test doc_to_target returning list for multiple targets."""
        task = Mock(spec=Task)
        task.features = []
        task.config = Mock()
        task.config.doc_to_target = ["{{answer1}}", "{{answer2}}"]
        task._config = task.config

        doc = {"answer1": "first", "answer2": "second"}

        result = Task.doc_to_target(task, doc)
        assert result == ["first", "second"]

    def test_override_doc_to_target(self):
        """Test overriding doc_to_target with parameter."""
        task = Mock(spec=Task)
        task.features = []
        task.config = Mock()
        task.config.doc_to_target = "default"
        task._config = task.config

        doc = {"answer": "test"}

        result = Task.doc_to_target(task, doc, doc_to_target="override")
        assert result == "override"

    def test_doc_to_target_type_error(self):
        """Test doc_to_target raises TypeError for invalid type."""
        task = Mock(spec=Task)
        task.config = Mock()
        task.config.doc_to_target = {"invalid": "type"}
        task._config = task.config

        doc = {"answer": "test"}

        with pytest.raises(TypeError):
            Task.doc_to_target(task, doc)

    def test_doc_to_target_literal_eval_edge_cases(self):
        """Test doc_to_target with edge cases for literal_eval."""
        task = Mock(spec=Task)
        task.features = []
        task.config = Mock()
        task.config.doc_to_choice = ["A", "B", "C"]
        task._config = task.config

        # Test numeric string conversion
        task.config.doc_to_target = "{{label}}"
        doc = {"label": "2"}
        result = Task.doc_to_target(task, doc)
        assert result == 2

        # Test non-numeric string stays as string
        doc = {"label": "abc"}
        result = Task.doc_to_target(task, doc)
        assert result == "abc"

        # Test mixed alphanumeric stays as string
        doc = {"label": "2a"}
        result = Task.doc_to_target(task, doc)
        assert result == "2a"


class TestDocToChoiceMethod:
    """Test suite for doc_to_choice method."""

    def test_doc_to_choice_with_field(self):
        """Test doc_to_choice when config points to a field name."""
        task = Mock(spec=Task)
        task.features = ["choices"]
        task.config = Mock()
        task.config.doc_to_choice = "choices"

        doc = {"choices": ["A", "B", "C", "D"]}

        result = Task.doc_to_choice(task, doc)
        assert result == ["A", "B", "C", "D"]

    def test_doc_to_choice_with_jinja_list(self):
        """Test doc_to_choice with Jinja template returning list as string."""
        task = Mock(spec=Task)
        task.features = []
        task.config = Mock()
        task.config.doc_to_choice = "{{choices}}"

        doc = {"choices": ["opt1", "opt2", "opt3"]}

        # The Jinja template will render the list as a string
        result = Task.doc_to_choice(task, doc)
        assert result == ["opt1", "opt2", "opt3"]

    def test_doc_to_choice_with_jinja_list_literal(self):
        """Test doc_to_choice with Jinja template creating a list literal."""
        task = Mock(spec=Task)
        task.features = []
        task.config = Mock()
        task.config.doc_to_choice = "{{[correct, wrong]}}"

        doc = {"correct": "The right answer", "wrong": "The wrong answer"}

        # The Jinja template will create a list literal and render it as a string
        result = Task.doc_to_choice(task, doc)
        assert result == ["The right answer", "The wrong answer"]

        # Test with another variation
        task.config.doc_to_choice = "{{[option_a, option_b, option_c]}}"
        doc = {"option_a": "Choice A", "option_b": "Choice B", "option_c": "Choice C"}
        result = Task.doc_to_choice(task, doc)
        assert result == ["Choice A", "Choice B", "Choice C"]

    def test_doc_to_choice_with_list_of_templates(self):
        """Test doc_to_choice with list of Jinja templates."""
        task = Mock(spec=Task)
        task.features = []
        task.config = Mock()
        task.config.doc_to_choice = ["{{choice_a}}", "{{choice_b}}", "{{choice_c}}"]

        doc = {"choice_a": "Apple", "choice_b": "Banana", "choice_c": "Cherry"}

        result = Task.doc_to_choice(task, doc)
        assert result == ["Apple", "Banana", "Cherry"]

    def test_doc_to_choice_with_dict(self):
        """Test doc_to_choice with dictionary config."""
        task = Mock(spec=Task)
        task.config = Mock()
        task.config.doc_to_choice = {
            "A": "First option",
            "B": "Second option",
            "C": "Third option",
        }

        doc = {}

        result = Task.doc_to_choice(task, doc)
        assert result == ["First option", "Second option", "Third option"]

    def test_doc_to_choice_with_callable(self):
        """Test doc_to_choice with a callable function."""

        def custom_choice_func(doc):
            return [f"Option {i}" for i in range(doc["num_choices"])]

        task = Mock(spec=Task)
        task.config = Mock()
        task.config.doc_to_choice = custom_choice_func

        doc = {"num_choices": 3}

        result = Task.doc_to_choice(task, doc)
        assert result == ["Option 0", "Option 1", "Option 2"]

    def test_doc_to_choice_none_error(self):
        """Test doc_to_choice logs error when not configured."""
        task = Mock(spec=Task)
        task.config = Mock()
        task.config.doc_to_choice = None

        doc = {}

        # When doc_to_choice is None, it logs an error and then raises TypeError
        with patch("lm_eval.api.task.eval_logger.error") as mock_error:
            with pytest.raises(TypeError):
                Task.doc_to_choice(task, doc)
            mock_error.assert_called_once_with(
                "doc_to_choice was called but not set in config"
            )

    def test_doc_to_choice_with_conditional(self):
        """Test doc_to_choice with Jinja conditional."""
        task = Mock(spec=Task)
        task.features = []
        task.config = Mock()
        task.config.doc_to_choice = "{{choices if has_choices else default_choices}}"

        doc = {
            "has_choices": True,
            "choices": ["A", "B"],
            "default_choices": ["X", "Y"],
        }

        result = Task.doc_to_choice(task, doc)
        assert result == ["A", "B"]

    def test_override_doc_to_choice(self):
        """Test overriding doc_to_choice with parameter."""
        task = Mock(spec=Task)
        task.config = Mock()
        task.config.doc_to_choice = ["A", "B"]

        doc = {}

        result = Task.doc_to_choice(task, doc, doc_to_choice=["X", "Y", "Z"])
        assert result == ["X", "Y", "Z"]

    def test_doc_to_choice_type_error(self):
        """Test doc_to_choice raises TypeError for invalid type."""
        task = Mock(spec=Task)
        task.config = Mock()
        task.config.doc_to_choice = 123  # Invalid type

        doc = {}

        with pytest.raises(TypeError):
            Task.doc_to_choice(task, doc)
