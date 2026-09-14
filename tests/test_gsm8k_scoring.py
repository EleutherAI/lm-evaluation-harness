"""Score-path tests for gsm8k, driven by the task's own declared config.

Four test modules build gsm8k `generate_until` requests, but three of them skip
in a standard CPU environment: `test_vllm.py` calls `importorskip("vllm")`,
`test_sglang.py` is `skipif(not torch.cuda.is_available())`, and
`test_hf_steered.py` needs an optional extra. That leaves
`tests/models/test_huggingface.py::Test_HFLM::test_generate_until` as the only
one that actually runs, and it asserts generated text, so it covers prompt
assembly and generation and stops before scoring. Extraction, normalization and
the metric have no equivalent, which means `gsm8k.yaml` can be edited in ways
that change every reported score with the suite still green.

These tests close that gap. Each one reads the values out of the task's declared
config rather than restating them, so a change to `gsm8k.yaml` reaches the
assertion. Restating the patterns here would make the tests immune to exactly
the changes they exist to catch. `tests/test_filters.py` does not cover this
either: it exercises `MultiChoiceRegexFilter`, which subclasses `RegexFilter`
but overrides `apply`, with patterns it defines itself, so neither
`RegexFilter.apply` nor gsm8k's declared patterns are reached by it.

No dataset, no model and no network: the config is read from the task index and
the filters are built through the public `build_filter_ensemble`, so nothing
here downloads gsm8k.
"""

import pytest

from lm_eval.api.instance import Instance
from lm_eval.api.metrics import exact_match_hf_evaluate
from lm_eval.filters import build_filter_ensemble
from lm_eval.tasks import TaskManager

# `load_yaml` is the loader TaskManager itself uses for a task's YAML
# (see `_factory._load_full_config`). It is reached directly here so the config
# can be read without instantiating the task, which would download the dataset.
# Happy to switch to a public accessor if you would rather expose one.
from lm_eval.tasks._yaml_loader import load_yaml


@pytest.fixture(scope="module")
def gsm8k_config():
    """gsm8k's declared config, without downloading the dataset."""
    entry = TaskManager().task_index.get("gsm8k")
    assert entry is not None and entry.yaml_path, "gsm8k is not in the task index"
    return load_yaml(entry.yaml_path)


@pytest.fixture(scope="module")
def metric_kwargs(gsm8k_config):
    """The kwargs gsm8k declares for `exact_match`."""
    metric = next(
        m for m in gsm8k_config["metric_list"] if m["metric"] == "exact_match"
    )
    return {
        key: value
        for key, value in metric.items()
        if key not in ("metric", "aggregation", "higher_is_better")
    }


def filter_output(config, name, model_output):
    """Run one model output through a declared filter pipeline, as the task does."""
    entry = next(f for f in config["filter_list"] if f["name"] == name)
    components = [
        (step["function"], {k: v for k, v in step.items() if k != "function"})
        for step in entry["filter"]
    ]
    instance = Instance(request_type="generate_until", doc={}, arguments=("",), idx=0)
    instance.resps = [model_output]
    build_filter_ensemble(name, components).apply([instance])
    return instance.filtered_resps[name]


def score(kwargs, prediction, reference, **overrides):
    """Score one pair through gsm8k's declared metric kwargs."""
    merged = {**kwargs, **overrides}
    return exact_match_hf_evaluate(
        predictions=[prediction], references=[reference], **merged
    )["exact_match"]


def test_strict_match_requires_the_answer_marker(gsm8k_config):
    """strict-match must extract the marked answer, not the first number.

    The `#### ` anchor in the declared pattern is what makes strict-match
    strict. Without it the filter takes the first number anywhere in the
    output, so a model that shows its working is scored against its own
    intermediate arithmetic.
    """
    assert (
        filter_output(gsm8k_config, "strict-match", "3 plus 7 is 10. #### 10") == "10"
    )


def test_flexible_extract_takes_the_models_final_answer(gsm8k_config):
    """flexible-extract must take the last match, not the first.

    It declares `group_select: -1`, which is the point of a flexible
    extractor: a model that corrects itself should be scored on the
    correction. Taking the first match scores the mistake instead.
    """
    output = "#### 3\nwait no\n#### 10"
    assert filter_output(gsm8k_config, "flexible-extract", output) == "10"


# One pair per declared normalization rule, chosen so the pair scores 1.0 with
# the full rule list and 0.0 with that single rule removed. Keyed by the rule
# exactly as it appears in the config.
NORMALIZATION_CASES = {
    ",": ("1234", "#### 1,234", "gsm8k golds carry thousands separators"),
    "\\$": (
        "$1,234",
        "#### 1,234",
        "models emit a leading dollar sign on money answers",
    ),
    "(?s).*#### ": ("72", "reasoning #### 72", "the gold's reasoning must be stripped"),
    "\\.$": ("72.", "#### 72", "a model ending a sentence emits a trailing period"),
}


def test_every_declared_normalization_rule_has_a_case(metric_kwargs):
    """The rules pinned below must be exactly the rules gsm8k declares.

    Without this, the parametrised test below is keyed off a list restated in
    this file, so a rule ADDED to `gsm8k.yaml` would be silently unpinned while
    the test named "every declared rule" went on passing. An added rule is not a
    hypothetical: adding `-` to `regexes_to_ignore` makes the prediction `-5`
    score 1.0 against the gold `5`, turning every sign error into a correct
    answer.
    """
    declared = list(metric_kwargs["regexes_to_ignore"])
    assert declared == list(NORMALIZATION_CASES), (
        "gsm8k's declared regexes_to_ignore no longer matches the rules pinned "
        f"here.\n  declared: {declared}\n  pinned:   {list(NORMALIZATION_CASES)}\n"
        "Add a case for each new rule, showing a pair whose score it changes."
    )


@pytest.mark.parametrize("rule", list(NORMALIZATION_CASES))
def test_every_declared_normalization_rule_is_load_bearing(rule, metric_kwargs):
    """Each entry in `regexes_to_ignore` must change a score it is meant to fix.

    No test currently names `regexes_to_ignore`, so any entry can be removed
    without the suite noticing, and each removal silently lowers reported
    accuracy for a whole class of correct answers.
    """
    declared = metric_kwargs["regexes_to_ignore"]
    assert rule in declared, f"{rule!r} is missing from the declared rules {declared}"

    prediction, reference, why = NORMALIZATION_CASES[rule]
    assert score(metric_kwargs, prediction, reference) == 1.0, why
    without_rule = [r for r in declared if r != rule]
    assert (
        score(metric_kwargs, prediction, reference, regexes_to_ignore=without_rule)
        == 0.0
    ), f"{rule!r} is declared but does not affect the score, so it is not pinned"


def test_punctuation_is_not_ignored_when_scoring(metric_kwargs):
    r"""gsm8k declares `ignore_punctuation: false`, and that must hold.

    Under the declared setting the prediction `72..` keeps a period after the
    `\.$` rule strips one, so it does not equal the gold. Turning the flag on
    strips both and the pair matches, which silently raises reported accuracy.
    """
    assert metric_kwargs["ignore_punctuation"] is False
    assert score(metric_kwargs, "72..", "#### 72") == 0.0
    assert score(metric_kwargs, "72..", "#### 72", ignore_punctuation=True) == 1.0


def test_a_correct_generation_scores_one_through_both_variants(
    gsm8k_config, metric_kwargs
):
    """End to end over the score path: raw generation to reported score.

    This is the structural counterpart to `test_generate_until`. That test
    stops at the generated text; this one starts there and runs the declared
    filters and the declared metric, which is the half currently untested.
    """
    generation = "Let me think. 3 plus 7 is 10.\n#### 10"
    gold = "#### 10"
    # Every DECLARED variant, not a restated pair, so a filter added to
    # filter_list is exercised rather than quietly skipped.
    variants = [f["name"] for f in gsm8k_config["filter_list"]]
    assert variants, "gsm8k declares no filters"
    for variant in variants:
        extracted = filter_output(gsm8k_config, variant, generation)
        assert score(metric_kwargs, extracted, gold) == 1.0, variant
