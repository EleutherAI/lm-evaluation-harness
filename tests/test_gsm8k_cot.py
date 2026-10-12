from pathlib import Path

import pytest

from lm_eval.filters.extraction import RegexFilter
from lm_eval.tasks._yaml_loader import load_yaml


TASKS_DIR = Path(__file__).parents[1] / "lm_eval" / "tasks"

COT_CONFIGS = [
    pytest.param(TASKS_DIR / "gsm8k" / "gsm8k-cot.yaml", id="gsm8k_cot"),
    pytest.param(
        TASKS_DIR / "gsm8k" / "gsm8k-cot-zeroshot.yaml", id="gsm8k_cot_zeroshot"
    ),
    pytest.param(
        TASKS_DIR / "gsm8k_platinum" / "gsm8k-platinum-cot.yaml",
        id="gsm8k_platinum_cot",
    ),
    pytest.param(
        TASKS_DIR / "gsm8k_platinum" / "gsm8k-platinum-cot-zeroshot.yaml",
        id="gsm8k_platinum_cot_zeroshot",
    ),
]


def strict_match(config_path, response):
    config = load_yaml(config_path, resolve_func=False)
    strict = next(f for f in config["filter_list"] if f["name"] == "strict-match")
    regex = next(step for step in strict["filter"] if step["function"] == "regex")
    filtered = RegexFilter(regex_pattern=regex["regex_pattern"]).apply(
        [[response]], [{}]
    )
    return filtered[0][0]


@pytest.mark.parametrize("config_path", COT_CONFIGS)
@pytest.mark.parametrize(
    "response, expected",
    [
        ("So she has 18 left. The answer is 18.", "18"),
        ("So she has 18 left. The answer is 18", "18"),
        ("So she has 18 left. The answer is 18\n\n", "18"),
        ("The answer is 7", "7"),
        ("The answer is 1,250\n", "1,250"),
        ("The answer is -3.5.", "-3.5"),
        ("I am not sure.", "[invalid]"),
    ],
)
def test_strict_match_keeps_the_whole_answer(config_path, response, expected):
    assert strict_match(config_path, response) == expected
