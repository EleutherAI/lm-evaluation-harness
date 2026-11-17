import pytest

from lm_eval.filters.extraction import RegexFilter
from lm_eval.filters.selection import TakeKFilter
from lm_eval.filters.transformation import LowercaseFilter
from lm_eval.tasks.simple_cooccurrence_bias.utils import process_results_gen


@pytest.mark.parametrize(
    "responses, expected",
    [
        (["man"] * 10, (10, 1, 0, 0)),
        (["man"] * 9 + ["woman"], (8, 1, 0, 0)),
        (["female"] * 8 + ["male"] * 2, (-6, 0, 1, 0)),
        (["man"] * 5 + ["woman"] * 5, (0, 0, 0, 0)),
        (["[invalid]"] * 10, (0, 0, 0, 1)),
    ],
)
def test_process_results_gen_counts_every_sample(responses, expected):
    result = process_results_gen({}, [responses])
    assert (
        result["difference_male_female"],
        result["pct_male_preferred"],
        result["pct_female_preferred"],
        result["pct_invalid"],
    ) == expected


def test_process_results_gen_with_task_filters():
    raw = [[" man who", " Man and", " woman", " man.", " man", " man,"] * 2]
    docs = [{}]
    resps = LowercaseFilter().apply(raw, docs)
    resps = RegexFilter(regex_pattern="(?=(female|woman|man|male))").apply(resps, docs)
    resps = list(TakeKFilter(k=10).apply(resps, docs))

    result = process_results_gen({}, resps)

    assert result["pct_male_preferred"] == 1
    assert result["pct_female_preferred"] == 0
    assert result["difference_male_female"] == 6
