import pytest

from lm_eval.filters.extraction import MultiChoiceRegexFilter


@pytest.mark.parametrize(
    "normalization",
    [
        {},
        {"ignore_case": True},
        {"ignore_punctuation": True},
        {"regexes_to_ignore": [r"\s+"]},
        {
            "ignore_case": True,
            "ignore_punctuation": True,
            "regexes_to_ignore": [r"^Answer:\s*"],
        },
    ],
)
def test_multichoice_nontext_responses_use_fallback_without_losing_alignment(
    normalization,
):
    filt = MultiChoiceRegexFilter(
        regex_pattern=r"(\([A-Z]\))",
        fallback="missing",
        **normalization,
    )
    resps = [[None, "(B)", 42, {"answer": "(A)"}, ["(A)"], ""]]
    docs = [{"choices": ["Alpha", "Beta"]}]

    assert filt.apply(resps, docs) == [
        ["missing", "(B)", "missing", "missing", "missing", "missing"]
    ]


def test_multichoice_text_fallback_paths_still_work_with_normalization():
    filt = MultiChoiceRegexFilter(
        regex_pattern=r"(\([A-Z]\))",
        ignore_case=True,
        ignore_punctuation=True,
        regexes_to_ignore=[r"^Answer:\s*"],
    )
    docs = [{"choices": ["Alpha!", "Beta?"]}]

    assert filt.apply([["Answer: BETA!", ": A", "(B)"]], docs) == [
        ["(B)", "(A)", "(B)"]
    ]
