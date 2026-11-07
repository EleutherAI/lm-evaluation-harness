"""Request construction for assistant prefixes selected from document columns."""

import random

import datasets
import pytest

from lm_eval.api.task import ConfigurableTask


def _chat_template(messages, add_generation_prompt=False):
    return "".join(
        f"<|{message['role']}|>{message['content']}" for message in messages
    ) + ("<|assistant|>" if add_generation_prompt else "")


def _make_task(prefix_field, prefix_value, delimiter=" ", multiple_input=False):
    docs = datasets.Dataset.from_list(
        [
            {
                "question": "Which choice?",
                "answer": "A" if multiple_input else 0,
                "prefix": prefix_value,
            }
        ]
    )
    return ConfigurableTask(
        config={
            "task": "request_prefix",
            "custom_dataset": lambda **kwargs: {"test": docs},
            "test_split": "test",
            "output_type": "multiple_choice",
            "doc_to_text": 0 if multiple_input else "question",
            "doc_to_target": "answer",
            "doc_to_choice": ["Left context", "Right context"]
            if multiple_input
            else ["A", "B"],
            "gen_prefix": prefix_field,
            "target_delimiter": delimiter,
            "num_fewshot": 0,
            "metric_list": [
                {"metric": "acc", "aggregation": "mean", "higher_is_better": True}
            ],
        }
    )


@pytest.mark.parametrize("prefix_value", ["Answer:", "Answer: ", "", "\n", None])
@pytest.mark.parametrize("delimiter", [" ", "::"])
@pytest.mark.parametrize("multiple_input", [False, True])
def test_chat_requests_match_equivalent_literal_column_value(
    prefix_value, delimiter, multiple_input
):
    """A prefix column must score the same strings as its resolved literal."""
    task = _make_task("prefix", prefix_value, delimiter, multiple_input)
    literal = _make_task(prefix_value, prefix_value, delimiter, multiple_input)

    for candidate in (task, literal):
        candidate.build_all_requests(
            apply_chat_template=True, chat_template=_chat_template
        )

    actual = [instance.arguments for instance in task.instances]
    expected = [instance.arguments for instance in literal.instances]
    assert actual == expected
    # Pin the intended separator independently of the literal control.
    assert actual[0][1] == (f"{delimiter}A" if prefix_value == "Answer:" else "A")


def test_template_prefix_is_rendered_once():
    """Choosing a separator must not render a potentially stateful template again."""
    task = _make_task("{{ ['Answer: ', 'Answer:'] | random }}", "unused")
    state = random.getstate()
    try:
        random.seed(42)
        expected_prefix = random.choice(["Answer: ", "Answer:"])
        expected_state = random.getstate()
        random.seed(42)

        task.build_all_requests(apply_chat_template=True, chat_template=_chat_template)

        assert random.getstate() == expected_state
        assert task.instances[0].arguments[0].endswith(expected_prefix)
        # Template-based separator behavior is deliberately unchanged.
        assert task.instances[0].arguments[1] == " A"
    finally:
        random.setstate(state)


@pytest.mark.parametrize("prefix_field", [None, ""])
def test_chat_requests_without_prefix_do_not_add_a_separator(prefix_field):
    task = _make_task(prefix_field, "unused", delimiter="::")
    task.build_all_requests(apply_chat_template=True, chat_template=_chat_template)

    assert [instance.arguments[1] for instance in task.instances] == ["A", "B"]


@pytest.mark.parametrize("prefix_field", ["prefix", "{{ prefix }}"])
def test_plain_text_requests_keep_the_configured_separator(prefix_field):
    task = _make_task(prefix_field, "Answer: ", delimiter="::")
    task.build_all_requests()

    assert [instance.arguments[1] for instance in task.instances] == ["::A", "::B"]
