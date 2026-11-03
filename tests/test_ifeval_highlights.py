import pytest


pytest.importorskip("langdetect")
pytest.importorskip("immutabledict")
pytest.importorskip("nltk")

from lm_eval.tasks.ifeval import instructions, utils


@pytest.mark.parametrize(
    ("response", "count"),
    [
        ("plain text", 0),
        ("**", 0),
        ("***", 0),
        ("****", 0),
        ("* * and ** ** and *** ***", 0),
        ("*one*", 1),
        ("**one**", 1),
        ("***one***", 1),
        ("*one* and **two**", 2),
        ("***one*** and ***two***", 2),
        ("***one*** and **two** and *three*", 3),
        ("*one**two*", 2),
        ("**one****two**", 2),
        ("*one**two**three*", 3),
        ("*one**two**three**four*", 4),
        ("***one*** *two**three**four*", 4),
        ("*one*\n***two***", 2),
        ("*one\ntwo*", 0),
    ],
)
def test_highlight_sections_count_each_recognized_span_once(response, count):
    checker = instructions.HighlightSectionChecker("highlight-test")
    # Test both sides of the threshold, including multiple overlapping spans.
    checker.build_description(num_highlights=count + 1)
    assert checker.check_following(response) is False
    if count:
        checker.build_description(num_highlights=count)
        assert checker.check_following(response) is True


@pytest.mark.parametrize(
    ("response", "minimum", "expected"),
    [
        ("***one***", 2, False),
        ("***one***", 1, True),
        ("***one*** and ***two***", 3, False),
        ("***one*** and ***two***", 2, True),
        ("*one* and **two**", 2, True),
        ("*one**two**three*", 3, True),
        ("*one**two**three**four*", 4, True),
        ("***one*** *two**three**four*", 4, True),
        ("plain text", 1, False),
        ("** **", 1, False),
    ],
)
def test_highlight_overlap_uses_actual_strict_and_loose_scoring(
    response, minimum, expected
):
    doc = {
        "key": 1,
        "prompt": f"Highlight at least {minimum} sections.",
        "instruction_id_list": ["detectable_format:number_highlighted_sections"],
        "kwargs": [{"num_highlights": minimum}],
    }
    assert utils.process_results(doc, [response]) == {
        "prompt_level_strict_acc": expected,
        "inst_level_strict_acc": [expected],
        "prompt_level_loose_acc": expected,
        "inst_level_loose_acc": [expected],
    }
