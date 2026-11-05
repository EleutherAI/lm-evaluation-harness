from pathlib import Path

import pytest

from lm_eval.filters import build_filter_ensemble
from lm_eval.tasks._yaml_loader import load_yaml


TASKS_DIR = Path(__file__).parent.parent / "lm_eval" / "tasks"
VARIANTS = ["zeroshot", "cot_zeroshot"]


def flexible_extract(variant: str, task: str, response: str) -> str:
    """Run the shipped `flexible-extract` filter list of a bbh task on one response."""
    cfg = load_yaml(TASKS_DIR / "bbh" / variant / f"{task}.yaml")
    (filter_cfg,) = [f for f in cfg["filter_list"] if f["name"] == "flexible-extract"]
    components = [
        [fn["function"], {k: v for k, v in fn.items() if k != "function"}]
        for fn in filter_cfg["filter"]
    ]
    resps = [[response]]
    for f in build_filter_ensemble("flexible-extract", components).filters:
        resps = f().apply(resps, [{}])
    return next(iter(resps))


# A final verdict that uses the task's own wording ("X lies", the exemplars'
# phrasing) must map to "no". Before, "lies" was not in either pattern, so the
# filter returned an earlier "tells the truth" about another person ("yes") or
# fell back to "[invalid]".
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize(
    "response",
    [
        "Leda lies.",
        "Leda is lying.",
        "Leda doesn't tell the truth.",
        "Leda doesn’t tell the truth.",
        "Leda isn't telling the truth.",
    ],
)
def test_web_of_lies_negative_verdict_maps_to_no(variant, response):
    assert flexible_extract(variant, "web_of_lies", response) == "no"


# Exemplar-style chains without the closing "So the answer is ..." line.
CHAIN_ENDS_LIES = (
    "(4) Fidel says Maybelle lies. Since we know from (3) that Maybelle tells "
    "the truth, if Fidel says Maybelle lies, then Fidel lies.\n"
    "(5) Leda says Fidel tells the truth. Since we know from (4) that Fidel "
    "lies, if Leda says Fidel tells the truth, then Leda lies.\n"
    "Now, the question asks: Does Leda tell the truth? We know from (5) that "
    "Leda lies."
)
CHAIN_ENDS_TRUTH = (
    "(5) Raymond says Millicent lies. Since we know from (4) that Millicent "
    "lies, if Raymond says Millicent lies, then Raymond tells the truth.\n"
    "Now, the question asks: Does Raymond tell the truth? We know from (5) "
    "that Raymond tells the truth."
)
# The question restated with "plausible", then an "implausible" verdict.
SPORTS_IMPLAUSIBLE = (
    'The question is whether "Santi Cazorla scored a touchdown." is '
    "plausible. Santi Cazorla is a soccer player and a touchdown is part of "
    "American football, so the sentence is implausible."
)


@pytest.mark.parametrize(
    "response, expected",
    [
        (CHAIN_ENDS_LIES, "no"),
        (CHAIN_ENDS_TRUTH, "yes"),
        # controls that already worked
        ("So the answer is No.", "no"),
        ("So the answer is Yes.", "yes"),
        ("Leda does not tell the truth.", "no"),
        ("Leda tells the truth.", "yes"),
    ],
)
def test_web_of_lies_cot_chain_takes_final_verdict(response, expected):
    assert flexible_extract("cot_zeroshot", "web_of_lies", response) == expected


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize(
    "response",
    [
        "Implausible.",
        "The sentence is implausible.",
        "That isn't plausible.",
        "That isn’t plausible.",
    ],
)
def test_sports_understanding_negative_verdict_maps_to_no(variant, response):
    assert flexible_extract(variant, "sports_understanding", response) == "no"


@pytest.mark.parametrize(
    "response, expected",
    [
        (SPORTS_IMPLAUSIBLE, "no"),
        # controls that already worked
        ("So the answer is no.", "no"),
        ("So the answer is yes.", "yes"),
        ("The sentence is not plausible.", "no"),
        ("The sentence is plausible.", "yes"),
    ],
)
def test_sports_understanding_cot_takes_final_verdict(response, expected):
    assert (
        flexible_extract("cot_zeroshot", "sports_understanding", response) == expected
    )


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("task", ["web_of_lies", "sports_understanding"])
def test_changed_tasks_are_versioned(variant, task):
    # flexible-extract scores of these tasks changed, so each task bumps its version
    cfg = load_yaml(TASKS_DIR / "bbh" / variant / f"{task}.yaml")
    assert cfg["metadata"]["version"] == 4.0
