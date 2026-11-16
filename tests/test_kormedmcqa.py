import pytest

from lm_eval.tasks.kormedmcqa.utils import ExtractChoiceFilter, extract_choice


@pytest.mark.parametrize(
    "output, expected",
    [
        (" B", "B"),
        ("B", "B"),
        ("b", "B"),
        ("정답: B", "B"),
        ("정답:B", "B"),
        ("정답 C", "C"),
        ("**정답: B**", "B"),
        ("정답은 D 입니다", "D"),
        ("정답은 D입니다", "D"),
        ("B.", "B"),
        ("The answer is (C).", "C"),
        # last match wins within a pattern, per the paper's reference code
        ("정답: B\n정답: C", "C"),
        # earlier patterns in the cascade take precedence over later ones
        ("A. foo\nB. bar\n정답: E", "E"),
        # no extractable answer
        ("잘 모르겠습니다", ""),
        ("", ""),
    ],
)
def test_kormedmcqa_extract_choice(output, expected):
    assert extract_choice(output) == expected


def test_kormedmcqa_extract_choice_filter():
    filt = ExtractChoiceFilter()

    resps = [["정답: C"], ["정답은 A 입니다"], ["모르겠어요"]]
    docs = [{}, {}, {}]

    assert filt.apply(resps, docs) == [["C"], ["A"], [""]]
