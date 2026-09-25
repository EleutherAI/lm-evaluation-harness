import pytest

from lm_eval.tasks.bbq.utils import (
    UNKNOWN_RESPONSES,
    doc_to_choice,
    doc_to_target,
    doc_to_targets,
)


@pytest.mark.parametrize("label", [0, 1, 2])
@pytest.mark.parametrize("unknown", UNKNOWN_RESPONSES)
def test_unknown_gold_targets_in_range_and_own_phrase_first(unknown, label):
    answers = ["The grandfather", "The grandson"]
    answers.insert(label, unknown)
    doc = {"ans0": answers[0], "ans1": answers[1], "ans2": answers[2], "label": label}

    choices = doc_to_choice(doc)
    targets = doc_to_targets(doc)

    assert all(0 <= t < len(choices) for t in targets)
    assert sorted(targets) == list(range(2, 2 + len(UNKNOWN_RESPONSES)))
    assert choices[doc_to_target(doc)] == unknown


def test_known_gold_single_target():
    doc = {
        "ans0": "Unknown",
        "ans1": "The grandfather",
        "ans2": "The grandson",
        "label": 2,
    }
    assert doc_to_targets(doc) == [doc_to_choice(doc).index("The grandson")]
