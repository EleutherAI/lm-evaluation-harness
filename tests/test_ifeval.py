"""Regression tests for IFEval checker-argument determinism (issue #4602).

The checkers fill missing instruction args from `random` at build time. Before
the fix those draws came from the process-wide RNG, so the verdict for a fixed
(prompt, response) pair depended on how many items had been scored before it,
and the same pair could score both True and False in one run.
"""

import importlib
import os
import random
import subprocess
import sys

import pytest

from lm_eval.tasks.ifeval import instructions
from lm_eval.tasks.ifeval.seeding import deterministic_checker_draws, stable_seed


SCORERS = [
    "lm_eval.tasks.ifeval.utils",
    # duplicate of the same scorer kept for the leaderboard task
    "lm_eval.tasks.leaderboard.ifeval.utils",
]

# Exactly three sentences. When the dataset leaves the checker args null the
# scorer fills them itself, so the verdict is whatever was drawn.
MISSING_ARGS_INP = {
    "key": 0,
    "instruction_id_list": ["length_constraints:number_sentences"],
    "prompt": "Write about the sea. Your response should contain at least 3 sentences.",
    "kwargs": [{"num_sentences": None, "relation": None}],
}
THREE_SENTENCES = "The sea is wide. The sea is deep. The sea is old."


def scorer_module(path):
    return importlib.import_module(path)


def score(module_path, inp_kwargs, response, variant="strict"):
    """Score one item from a given global-RNG position."""
    module = scorer_module(module_path)
    inp = module.InputExample(**inp_kwargs)
    fn = getattr(module, f"test_instruction_following_{variant}")
    return fn(inp, response)


@pytest.mark.parametrize("module_path", SCORERS)
@pytest.mark.parametrize("variant", ["strict", "loose"])
def test_missing_args_score_identically_from_any_rng_position(module_path, variant):
    verdicts = set()
    for trial in range(10):
        random.seed(trial)
        for _ in range(3 * trial):
            random.random()
        out = score(module_path, MISSING_ARGS_INP, THREE_SENTENCES, variant)
        verdicts.add((out.follow_all_instructions, tuple(out.follow_instruction_list)))
    assert len(verdicts) == 1, f"identical input scored {verdicts}"


@pytest.mark.parametrize("module_path", SCORERS)
def test_scoring_does_not_consume_global_random(module_path):
    random.seed(0)
    before = random.getstate()
    score(module_path, MISSING_ARGS_INP, THREE_SENTENCES, "strict")
    score(module_path, MISSING_ARGS_INP, THREE_SENTENCES, "loose")
    assert random.getstate() == before


@pytest.mark.parametrize("module_path", SCORERS)
def test_present_args_are_unaffected(module_path):
    # All args present: nothing is filled, so nothing is drawn either.
    follows = {
        "key": 1,
        "instruction_id_list": ["length_constraints:number_sentences"],
        "prompt": "Write about the sea.",
        "kwargs": [{"num_sentences": 3, "relation": "at least"}],
    }
    random.seed(0)
    before = random.getstate()
    assert score(module_path, follows, THREE_SENTENCES).follow_all_instructions
    assert random.getstate() == before

    too_short = dict(
        follows, key=2, kwargs=[{"num_sentences": 5, "relation": "at least"}]
    )
    assert not score(module_path, too_short, THREE_SENTENCES).follow_all_instructions


def test_seed_depends_on_item_content():
    def seed(prompt, kwargs=None):
        return stable_seed("length_constraints:number_sentences", prompt, kwargs or {})

    assert seed("Prompt A.") != seed("Prompt B.")
    assert seed("Prompt A.") == seed("Prompt A.")
    assert seed("Prompt A.", {"num_sentences": 3}) != seed("Prompt A.", {})


def test_seed_is_stable_across_processes():
    # `hash` is salted per process, so the seed must not be built from it.
    code = (
        "from lm_eval.tasks.ifeval.seeding import stable_seed;"
        "print(stable_seed('length_constraints:number_sentences', 'a prompt', {}))"
    )
    seeds = set()
    for salt in ("0", "1"):
        env = dict(os.environ, PYTHONHASHSEED=salt)
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        seeds.add(result.stdout.strip())
    assert len(seeds) == 1


def test_error_while_building_restores_random():
    bad_relation = {
        "key": 3,
        "instruction_id_list": ["length_constraints:number_sentences"],
        "prompt": "Write about the sea.",
        "kwargs": [{"num_sentences": None, "relation": "approximately"}],
    }
    random.seed(0)
    before = random.getstate()
    with pytest.raises(ValueError):
        score("lm_eval.tasks.ifeval.utils", bad_relation, THREE_SENTENCES)
    assert random.getstate() == before
    assert instructions.random is random


def test_helper_restores_module_attribute():
    original = instructions.random
    with deterministic_checker_draws(0, instructions):
        assert instructions.random is not original
        first = instructions.random.randint(0, 10**6)
    assert instructions.random is original
    with deterministic_checker_draws(0, instructions):
        assert instructions.random.randint(0, 10**6) == first
