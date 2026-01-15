"""Tests for fewshot context formatting (build_qa_turn and fewshot_context)."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from lm_eval.api.task import ConfigurableTask
from lm_eval.api.utils import Message, maybe_delimit, multiturn_to_singleturn


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_delimiters():
    """Default delimiters used in tasks."""
    return {"tgt_delim": " ", "few_delim": "\n\n"}


# =============================================================================
# Message Tests
# =============================================================================


class TestMessage:
    """Tests for the Message dataclass."""

    def test_to_dict_excludes_private_fields(self):
        """to_dict excludes fields starting with underscore."""
        msg = Message("user", "Hello", _delimiter="\n\n")

        result = msg.to_dict()

        assert result == {"role": "user", "content": "Hello"}
        assert "_delimiter" not in result

    def test_to_text_appends_delimiter(self):
        """to_text returns content + delimiter."""
        msg = Message("user", "Hello", "\n\n")

        result = msg.to_text()

        assert result == "Hello\n\n"

    def test_to_text_empty_delimiter(self):
        """to_text with empty delimiter returns just content."""
        msg = Message("assistant", "Answer")

        result = msg.to_text()

        assert result == "Answer"


# =============================================================================
# maybe_delimit Tests
# =============================================================================


class TestMaybeDelimit:
    """Tests for the maybe_delimit helper function."""

    def test_both_present_no_whitespace(self):
        """Adds delimiter when neither has whitespace at boundary."""
        result = maybe_delimit("prefix", "suffix", " ")

        assert result == "prefix suffix"

    def test_prefix_ends_with_space(self):
        """No extra delimiter when prefix ends with space."""
        result = maybe_delimit("prefix ", "suffix", "SPACE")

        assert result == "prefix suffix"

    def test_suffix_starts_with_space(self):
        """No extra delimiter when suffix starts with space."""
        result = maybe_delimit("prefix", " suffix", "DELIM")

        assert result == "prefix suffix"

    def test_both_have_whitespace(self):
        """No delimiter added when both have whitespace."""
        result = maybe_delimit("prefix ", " suffix", "X")

        assert result == "prefix  suffix"

    def test_prefix_only(self):
        """Returns prefix when suffix is None/empty."""
        assert maybe_delimit("prefix", None) == "prefix"
        assert maybe_delimit("prefix", "") == "prefix"

    def test_suffix_only(self):
        """Returns suffix when prefix is None/empty."""
        assert maybe_delimit(None, "suffix") == "suffix"
        assert maybe_delimit("", "suffix") == "suffix"

    def test_both_empty(self):
        """Returns empty string when both None/empty."""
        assert maybe_delimit(None, None) == ""
        assert maybe_delimit("", "") == ""

    def test_custom_delimiter(self):
        """Uses custom delimiter."""
        result = maybe_delimit("a", "b", "---")

        assert result == "a---b"


# =============================================================================
# multiturn_to_singleturn Tests
# =============================================================================


class TestMultiturnToSingleturn:
    """Tests for collapsing multiturn into single user message."""

    def test_collapses_user_messages(self):
        """Multiple user messages collapse into one."""
        messages = [
            Message("user", "Q1", " "),
            Message("assistant", "A1", "\n\n"),
            Message("user", "Q2", ""),
        ]

        result = multiturn_to_singleturn(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Q1 A1\n\nQ2"

    def test_preserves_final_assistant(self):
        """Final assistant message is kept separate."""
        messages = [
            Message("user", "Q1", " "),
            Message("assistant", "A1", "\n\n"),
            Message("user", "Q2", " "),
            Message("assistant", "Final"),
        ]

        result = multiturn_to_singleturn(messages)

        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "Final"

    def test_preserves_system_message(self):
        """System message stays separate at the front."""
        messages = [
            Message("system", "You are helpful", ""),
            Message("user", "Question", ""),
        ]

        result = multiturn_to_singleturn(messages)

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful"
        assert result[1]["role"] == "user"

    def test_system_with_assistant_ending(self):
        """System + collapsed user + final assistant."""
        messages = [
            Message("system", "System", ""),
            Message("user", "Q", " "),
            Message("assistant", "A", ""),
        ]

        result = multiturn_to_singleturn(messages)

        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"


# =============================================================================
# build_qa_turn Tests
# =============================================================================


def messages_to_text(msgs: list[Message]) -> str:
    """Helper to convert messages to text."""
    return "".join(m.to_text() for m in msgs)


class TestBuildQaTurn:
    """Tests for ConfigurableTask.build_qa_turn method."""

    @pytest.fixture
    def task(self):
        """Create a mock task to call build_qa_turn on."""
        return Mock(spec=ConfigurableTask)

    def test_basic_qa_format(self, task):
        """Basic question-answer produces user + assistant messages."""
        msgs = ConfigurableTask.build_qa_turn(
            task, q="Question?", a="Answer", tgt_delim=" ", few_delim="\n\n"
        )

        assert len(msgs) == 2
        assert msgs[0].role == "user"
        assert msgs[0].content == "Question?"
        assert msgs[1].role == "assistant"
        assert msgs[1].content == "Answer"
        assert messages_to_text(msgs) == "Question? Answer\n\n"

    def test_no_answer_format(self, task):
        """Without answer, only user message with no delimiter."""
        msgs = ConfigurableTask.build_qa_turn(task, q="Question?")

        assert len(msgs) == 1
        assert msgs[0].role == "user"
        assert msgs[0].content == "Question?"
        assert messages_to_text(msgs) == "Question?"

    def test_choice_with_int_answer(self, task):
        """Answer as index into choices list."""
        msgs = ConfigurableTask.build_qa_turn(
            task,
            q="Pick one:",
            c=["Apple", "Banana", "Cherry"],
            a=1,
            tgt_delim=" ",
            few_delim="\n\n",
        )

        assert len(msgs) == 2
        assert msgs[1].content == "Banana"
        assert messages_to_text(msgs) == "Pick one: Banana\n\n"

    def test_answer_as_string_directly(self, task):
        """Answer provided directly as string."""
        msgs = ConfigurableTask.build_qa_turn(
            task, q="What is 2+2?", a="4", tgt_delim=" ", few_delim="\n\n"
        )

        assert msgs[1].content == "4"
        assert messages_to_text(msgs) == "What is 2+2? 4\n\n"

    def test_answer_as_list(self, task):
        """Answer as list takes first element (multiple targets)."""
        msgs = ConfigurableTask.build_qa_turn(
            task, q="Question?", a=["first", "second"], tgt_delim=" ", few_delim="\n\n"
        )

        assert msgs[1].content == "first"

    def test_gen_prefix_without_answer(self, task):
        """gen_prefix adds assistant message when no answer."""
        msgs = ConfigurableTask.build_qa_turn(
            task, q="Question?", gen_prefix="Let me think"
        )

        assert len(msgs) == 2
        assert msgs[0].role == "user"
        assert msgs[1].role == "assistant"
        assert msgs[1].content == "Let me think"
        assert messages_to_text(msgs) == "Question? Let me think"

    def test_gen_prefix_with_answer(self, task):
        """gen_prefix prepended to answer (for fewshot examples)."""
        msgs = ConfigurableTask.build_qa_turn(
            task,
            q="Question?",
            a="Answer",
            gen_prefix="Think:",
            tgt_delim=" ",
            few_delim="\n\n",
        )

        # Only 2 messages: gen_prefix is prepended to answer, not added separately
        assert len(msgs) == 2
        assert msgs[0].role == "user"
        assert msgs[1].role == "assistant"
        assert msgs[1].content == "Think: Answer"
        assert messages_to_text(msgs) == "Question? Think: Answer\n\n"

    def test_gen_prefix_spacing_added_when_needed(self, task):
        """Space added between gen_prefix and answer when neither has whitespace."""
        msgs = ConfigurableTask.build_qa_turn(
            task, q="Q", a="Answer", gen_prefix="Prefix:", tgt_delim=" "
        )

        assert msgs[1].content == "Prefix: Answer"

    def test_gen_prefix_no_extra_space_when_prefix_has_trailing(self, task):
        """No extra space when gen_prefix ends with whitespace."""
        msgs = ConfigurableTask.build_qa_turn(
            task, q="Q", a="Answer", gen_prefix="Prefix: ", tgt_delim=" "
        )

        assert msgs[1].content == "Prefix: Answer"

    def test_gen_prefix_no_extra_space_when_answer_has_leading(self, task):
        """No extra space when answer starts with whitespace."""
        msgs = ConfigurableTask.build_qa_turn(
            task, q="Q", a=" Answer", gen_prefix="Prefix:", tgt_delim=" "
        )

        assert msgs[1].content == "Prefix: Answer"

    def test_gen_prefix_without_answer_preserves_content(self, task):
        """gen_prefix used as-is when no answer (target question)."""
        msgs = ConfigurableTask.build_qa_turn(
            task,
            q="Q",
            gen_prefix="The answer is:",
            tgt_delim=" ",
        )

        assert len(msgs) == 2
        assert msgs[1].role == "assistant"
        assert msgs[1].content == "The answer is:"

    def test_gen_prefix_with_trailing_space_without_answer(self, task):
        """gen_prefix with trailing space preserved when no answer."""
        msgs = ConfigurableTask.build_qa_turn(
            task, q="Q", gen_prefix="Answer: ", tgt_delim=" "
        )

        assert msgs[1].content == "Answer: "

    def test_custom_delimiters(self, task):
        """Custom delimiters are respected."""
        msgs = ConfigurableTask.build_qa_turn(
            task, q="Q", a="A", tgt_delim="->", few_delim="||"
        )

        assert messages_to_text(msgs) == "Q->A||"

    def test_empty_delimiters(self, task):
        """Empty delimiters produce no spacing."""
        msgs = ConfigurableTask.build_qa_turn(
            task, q="Q", a="A", tgt_delim="", few_delim=""
        )

        assert messages_to_text(msgs) == "QA"

    def test_whitespace_delimiter_matrix(self, task):
        r"""Whitespace interaction matrix for build_qa_turn.

        Two boundaries are checked:
        1. Q↔P: requires_delimiter(q, gen_prefix) → determines user message's _delimiter
        2. P↔A: maybe_delimit(gen_prefix, answer, delimiter=" ") → determines spacing in assistant content
        (note: always adds space after gen_prefix, if delimiter required)

        | q      | gen_prefix | a      | messages                | text         |
        |--------|------------|--------|-------------------------|--------------|
        | "Q"    | None       | "A"    | [U:"Q", A:"A"]          | "QXA"        |
        | "Q"    | None       | None   | [U:"Q"]                 | "Q"          |
        | "Q"    | "P"        | None   | [U:"Q", A:"P"]          | "QXP"        |
        | "Q"    | "P"        | "A"    | [U:"Q", A:"P A"]        | "QXP A"      |
        | "Q\n"  | "P"        | "A"    | [U:"Q\n", A:"P A"]      | "Q\nP A"     |
        | "Q"    | "\nP"      | None   | [U:"Q", A:"\nP"]        | "Q\nP"       |
        | "Q"    | "P\n"      | "A"    | [U:"Q", A:"P\nA"]       | "QXP\nA"     |
        | "Q"    | "P"        | "\nA"  | [U:"Q", A:"P\nA"]       | "QXP\nA"     |

        U = user message, A = assistant message
        X = tgt_delim (the delimiter between Q and P/A when needed)
        """
        # Row 1: "Q" + None + "A" → "QXA"
        msgs = ConfigurableTask.build_qa_turn(
            task, q="Q", a="A", tgt_delim="X", few_delim=""
        )
        assert msgs[0]._delimiter == "X"
        assert messages_to_text(msgs) == "QXA"

        # Row 2: "Q" + None + None → "Q"
        msgs = ConfigurableTask.build_qa_turn(task, q="Q")
        assert msgs[0]._delimiter == ""
        assert messages_to_text(msgs) == "Q"

        # Row 3: "Q" + "P" + None → "QXP"
        msgs = ConfigurableTask.build_qa_turn(
            task, q="Q", gen_prefix="P", tgt_delim="X"
        )
        assert msgs[0]._delimiter == "X"
        assert messages_to_text(msgs) == "QXP"

        # Row 4: "Q" + "P" + "A" → "QXP A"
        msgs = ConfigurableTask.build_qa_turn(
            task, q="Q", a="A", gen_prefix="P", tgt_delim="X", few_delim=""
        )
        assert msgs[0]._delimiter == "X"
        assert messages_to_text(msgs) == "QXP A"

        # Row 5: "Q\n" + "P" + "A" → "Q\nP A" (q ends with \n, no X needed)
        msgs = ConfigurableTask.build_qa_turn(
            task, q="Q\n", a="A", gen_prefix="P", tgt_delim="X", few_delim=""
        )
        assert msgs[0]._delimiter == ""
        assert messages_to_text(msgs) == "Q\nP A"

        # Row 6: "Q" + "\nP" + None → "Q\nP" (P starts with \n, no X needed)
        msgs = ConfigurableTask.build_qa_turn(
            task, q="Q", gen_prefix="\nP", tgt_delim="X"
        )
        assert msgs[0]._delimiter == ""
        assert messages_to_text(msgs) == "Q\nP"

        # Row 7: "Q" + "P\n" + "A" → "QXP\nA" (P ends with \n, no extra space before A)
        msgs = ConfigurableTask.build_qa_turn(
            task, q="Q", a="A", gen_prefix="P\n", tgt_delim="X", few_delim=""
        )
        assert msgs[0]._delimiter == "X"
        assert messages_to_text(msgs) == "QXP\nA"

        # Row 8: "Q" + "P" + "\nA" → "QXP\nA" (A starts with \n, no extra space after P)
        msgs = ConfigurableTask.build_qa_turn(
            task, q="Q", a="\nA", gen_prefix="P", tgt_delim="X", few_delim=""
        )
        assert msgs[0]._delimiter == "X"
        assert messages_to_text(msgs) == "QXP\nA"

    def test_raises_on_non_string_question(self, task):
        """Raises AssertionError if question is not a string."""
        with pytest.raises(AssertionError, match="not a string"):
            ConfigurableTask.build_qa_turn(task, q=123, a="A")  # type: ignore

    def test_answer_index_zero_uses_delimiter(self, task):
        """Answer index 0 should still use target delimiter (regression test for #3452).

        When answer is an integer index into choices, a=0 should be treated as a valid
        answer, not as falsy. Previously, a=0 caused the target delimiter to be skipped.
        """
        choices = ["A", "B", "C", "D"]
        msgs = ConfigurableTask.build_qa_turn(
            task, q="Question?", c=choices, a=0, tgt_delim=" ", few_delim="\n\n"
        )

        # Should have 2 messages: user question with delimiter, assistant answer
        assert len(msgs) == 2
        assert msgs[0].role == "user"
        assert msgs[0]._delimiter == " "  # delimiter should be applied
        assert msgs[1].role == "assistant"
        assert msgs[1].content == "A"  # choices[0]
        assert messages_to_text(msgs) == "Question? A\n\n"

    def test_answer_index_nonzero_uses_delimiter(self, task):
        """Answer index > 0 should use target delimiter."""
        choices = ["A", "B", "C", "D"]
        msgs = ConfigurableTask.build_qa_turn(
            task, q="Question?", c=choices, a=2, tgt_delim=" ", few_delim="\n\n"
        )

        assert msgs[0]._delimiter == " "
        assert msgs[1].content == "C"  # choices[2]
        assert messages_to_text(msgs) == "Question? C\n\n"


# =============================================================================
# Fewshot Context Tests
# =============================================================================


class TestFewshotContext:
    """Tests for ConfigurableTask.fewshot_context method."""

    def test_zero_shot_format(self, mock_configurable_task):
        """Zero-shot: just the question."""
        mock_configurable_task.doc_to_text = Mock(
            return_value="What is the capital of France?"
        )
        mock_configurable_task.doc_to_target = Mock(return_value="Paris")

        result = ConfigurableTask.fewshot_context(
            mock_configurable_task, doc={"q": "test"}, num_fewshot=0
        )

        assert result == "What is the capital of France?"

    def test_one_shot_format(self, mock_configurable_task):
        """One-shot: one example + target question."""
        fewshot_doc = {"q": "What is 1+1?", "a": "2"}
        target_doc = {"q": "What is 2+2?", "a": "4"}

        mock_configurable_task.sampler.sample.return_value = [fewshot_doc]
        mock_configurable_task.doc_to_text = Mock(side_effect=lambda d, *args: d["q"])
        mock_configurable_task.doc_to_target = Mock(side_effect=lambda d, *args: d["a"])

        result = ConfigurableTask.fewshot_context(
            mock_configurable_task, doc=target_doc, num_fewshot=1
        )

        assert result == "What is 1+1? 2\n\nWhat is 2+2?"

    def test_two_shot_format(self, mock_configurable_task):
        """Two-shot: two examples + target question."""
        fs_docs = [
            {"q": "Q1", "a": "A1"},
            {"q": "Q2", "a": "A2"},
        ]
        target_doc = {"q": "Q3", "a": "A3"}

        mock_configurable_task.sampler.sample.return_value = fs_docs
        mock_configurable_task.doc_to_text = Mock(side_effect=lambda d, *args: d["q"])
        mock_configurable_task.doc_to_target = Mock(side_effect=lambda d, *args: d["a"])

        result = ConfigurableTask.fewshot_context(
            mock_configurable_task, doc=target_doc, num_fewshot=2
        )

        assert result == "Q1 A1\n\nQ2 A2\n\nQ3"

    def test_with_system_instruction(self, mock_configurable_task):
        """System instruction prepended to context."""
        mock_configurable_task.doc_to_text = Mock(return_value="Question?")
        mock_configurable_task.doc_to_target = Mock(return_value="Answer")

        system_instruction = "You are a helpful assistant.\n"

        result = ConfigurableTask.fewshot_context(
            mock_configurable_task,
            doc={},
            num_fewshot=0,
            system_instruction=system_instruction,
        )

        assert result == system_instruction + "Question?"

    def test_with_description(self, mock_configurable_task):
        """Description from config is included."""
        description = "Answer math questions.\n"
        mock_configurable_task.config.description = description
        mock_configurable_task.resolve_field = Mock(return_value=description)
        mock_configurable_task.doc_to_text = Mock(return_value="2+2?")
        mock_configurable_task.doc_to_target = Mock(return_value="4")

        result = ConfigurableTask.fewshot_context(
            mock_configurable_task, doc={}, num_fewshot=0
        )

        assert result == f"{description}2+2?"

    def test_system_instruction_and_description(self, mock_configurable_task):
        """System instruction combined with description."""
        description = "Answer math questions.\n"
        system_instruction = "Be helpful."
        mock_configurable_task.config.description = description
        mock_configurable_task.resolve_field = Mock(return_value=description)
        mock_configurable_task.doc_to_text = Mock(return_value="2+2?")
        mock_configurable_task.doc_to_target = Mock(return_value="4")

        result = ConfigurableTask.fewshot_context(
            mock_configurable_task,
            doc={},
            num_fewshot=0,
            system_instruction=system_instruction,
        )

        assert result == f"{system_instruction}\n\n{description}2+2?"

    def test_with_choices(self, mock_configurable_task):
        """Multiple choice with answer as index."""
        mock_configurable_task.config.doc_to_choice = "choices"
        mock_configurable_task.fewshot_cfg.doc_to_choice = "choices"

        fs_doc = {"q": "Pick:", "a": 0}
        target_doc = {"q": "Pick a fruit:", "a": 1}

        mock_configurable_task.sampler.sample.return_value = [fs_doc]
        mock_configurable_task.doc_to_text = Mock(side_effect=lambda d, *args: d["q"])
        mock_configurable_task.doc_to_target = Mock(side_effect=lambda d, *args: d["a"])
        mock_configurable_task.doc_to_choice = Mock(
            side_effect=lambda d, *args: ["A", "B"]
            if d == fs_doc
            else ["Apple", "Banana"]
        )

        result = ConfigurableTask.fewshot_context(
            mock_configurable_task, doc=target_doc, num_fewshot=1
        )

        # Fewshot uses choices[0]="A", target question only (no answer)
        assert "A\n\n" in result
        assert result.endswith("Pick a fruit:")

    def test_custom_delimiters(self, mock_configurable_task):
        """Custom delimiters are respected."""
        mock_configurable_task.config.target_delimiter = "->"
        mock_configurable_task.config.fewshot_delimiter = "||"
        mock_configurable_task.fewshot_cfg.target_delimiter = "->"
        mock_configurable_task.fewshot_cfg.fewshot_delimiter = "||"

        fs_doc = {"q": "Q1", "a": "A1"}
        target_doc = {"q": "Q2", "a": "A2"}
        mock_configurable_task.sampler.sample.return_value = [fs_doc]
        mock_configurable_task.doc_to_text = Mock(side_effect=lambda d, *args: d["q"])
        mock_configurable_task.doc_to_target = Mock(side_effect=lambda d, *args: d["a"])

        result = ConfigurableTask.fewshot_context(
            mock_configurable_task, doc=target_doc, num_fewshot=1
        )

        assert result == "Q1->A1||Q2"

    def test_gen_prefix_in_fewshot(self, mock_configurable_task):
        """gen_prefix from fewshot_cfg is applied to fewshot examples."""
        fs_doc = {"q": "Q1", "a": "A1"}
        target_doc = {"q": "Q2", "a": "A2"}
        mock_configurable_task.sampler.sample.return_value = [fs_doc]
        mock_configurable_task.doc_to_text = Mock(side_effect=lambda d, *args: d["q"])
        mock_configurable_task.doc_to_target = Mock(side_effect=lambda d, *args: d["a"])
        # fewshot examples use fewshot_cfg.gen_prefix
        mock_configurable_task.fewshot_cfg.gen_prefix = "Answer:"
        # resolve_field returns the gen_prefix value (not a template)
        mock_configurable_task.resolve_field = Mock(side_effect=lambda doc, val: val)

        result = ConfigurableTask.fewshot_context(
            mock_configurable_task, doc=target_doc, num_fewshot=1, gen_prefix="Answer:"
        )

        # Fewshot answer should have gen_prefix prepended
        assert "Answer: A1" in result
        # Target should end with gen_prefix
        assert result.endswith("Answer:")

    def test_sampler_excludes_eval_doc_when_same_split(self, mock_configurable_task):
        """When fewshot_split == test_split, eval_doc is passed to sampler."""
        mock_configurable_task.config.fewshot_split = "test"
        mock_configurable_task.config.test_split = "test"
        mock_configurable_task.fewshot_cfg.split = "test"
        mock_configurable_task.doc_to_text = Mock(return_value="Q")
        mock_configurable_task.doc_to_target = Mock(return_value="A")

        eval_doc = {"id": 123}
        ConfigurableTask.fewshot_context(
            mock_configurable_task, doc=eval_doc, num_fewshot=1
        )

        # Sampler should be called with eval_doc to exclude it
        mock_configurable_task.sampler.sample.assert_called_once_with(
            n=1, eval_doc=eval_doc
        )

    def test_sampler_no_exclusion_when_different_split(self, mock_configurable_task):
        """When fewshot_split != test_split, eval_doc is not passed to sampler."""
        mock_configurable_task.config.fewshot_split = "train"
        mock_configurable_task.config.test_split = "test"
        mock_configurable_task.fewshot_cfg.split = "train"
        mock_configurable_task.doc_to_text = Mock(return_value="Q")
        mock_configurable_task.doc_to_target = Mock(return_value="A")

        eval_doc = {"id": 123}
        ConfigurableTask.fewshot_context(
            mock_configurable_task, doc=eval_doc, num_fewshot=1
        )

        # Sampler should be called without eval_doc
        mock_configurable_task.sampler.sample.assert_called_once_with(
            n=1, eval_doc=None
        )

    def test_chat_template_multiturn(self, mock_configurable_task):
        """Chat template with fewshot_as_multiturn=True keeps messages separate."""
        fs_doc = {"q": "Q1", "a": "A1"}
        target_doc = {"q": "Q2", "a": "A2"}
        mock_configurable_task.sampler.sample.return_value = [fs_doc]
        mock_configurable_task.doc_to_text = Mock(side_effect=lambda d, *args: d["q"])
        mock_configurable_task.doc_to_target = Mock(side_effect=lambda d, *args: d["a"])

        captured_messages = []

        def mock_chat_template(msgs, **kwargs):
            captured_messages.extend(msgs)
            return "<chat>"

        result = ConfigurableTask.fewshot_context(
            mock_configurable_task,
            doc=target_doc,
            num_fewshot=1,
            apply_chat_template=True,
            fewshot_as_multiturn=True,
            chat_template=mock_chat_template,
        )

        assert result == "<chat>"
        # Should have 3 separate messages: user, assistant, user
        assert len(captured_messages) == 3
        assert captured_messages[0]["role"] == "user"
        assert captured_messages[1]["role"] == "assistant"
        assert captured_messages[2]["role"] == "user"

    def test_chat_template_singleturn(self, mock_configurable_task):
        """Chat template with fewshot_as_multiturn=False collapses to single user."""
        fs_doc = {"q": "Q1", "a": "A1"}
        target_doc = {"q": "Q2", "a": "A2"}
        mock_configurable_task.sampler.sample.return_value = [fs_doc]
        mock_configurable_task.doc_to_text = Mock(side_effect=lambda d, *args: d["q"])
        mock_configurable_task.doc_to_target = Mock(side_effect=lambda d, *args: d["a"])

        captured_messages = []

        def mock_chat_template(msgs, **kwargs):
            captured_messages.extend(msgs)
            return "<chat>"

        result = ConfigurableTask.fewshot_context(
            mock_configurable_task,
            doc=target_doc,
            num_fewshot=1,
            apply_chat_template=True,
            fewshot_as_multiturn=False,
            chat_template=mock_chat_template,
        )

        assert result == "<chat>"
        # Should collapse to single user message
        assert len(captured_messages) == 1
        assert captured_messages[0]["role"] == "user"


# =============================================================================
# Chat Template Format Tests
# =============================================================================


class TestChatTemplateFormat:
    """Tests for chat template message list formatting."""

    def test_messages_to_dict_list(self):
        """Messages convert to list of dicts for chat template."""
        msgs = [
            Message("system", "Be helpful", ""),
            Message("user", "Question", ""),
            Message("assistant", "Answer", ""),
        ]

        result = [m.to_dict() for m in msgs]

        assert result == [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]

    def test_singleturn_collapse_for_chat(self):
        """Multiturn collapses correctly for chat template."""
        msgs = [
            Message("system", "System", ""),
            Message("user", "Q1", " "),
            Message("assistant", "A1", "\n\n"),
            Message("user", "Q2", ""),
        ]

        result = multiturn_to_singleturn(msgs)

        assert len(result) == 2
        assert result[0] == {"role": "system", "content": "System"}
        assert result[1]["role"] == "user"
        assert "Q1" in result[1]["content"]
        assert "A1" in result[1]["content"]
        assert "Q2" in result[1]["content"]
