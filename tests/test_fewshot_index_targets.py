import pytest

from lm_eval.api.task import ConfigurableTask


@pytest.mark.parametrize(
    "answer,expected",
    [
        ([0, 1], "Paris"),
        ([1, 0], "France"),
        (0, "Paris"),
        (1, "France"),
        (["Rome", "Italy"], "Rome"),
        ("Rome", "Rome"),
    ],
)
def test_qa_turn_resolves_first_target_before_choice_lookup(answer, expected):
    task = ConfigurableTask.__new__(ConfigurableTask)
    messages = task.build_qa_turn(q="Where?", c=["Paris", "France"], a=answer)
    assert messages[1].to_dict() == {"role": "assistant", "content": expected}
    assert expected in "".join(m.to_text() for m in messages)
