import json
import math
from pathlib import Path

import datasets
import numpy as np
import pytest

from lm_eval.api.model import LM
from lm_eval.api.task import ConfigurableTask
from lm_eval.evaluator import evaluate
from lm_eval.tasks._yaml_loader import load_yaml
from lm_eval.tasks.simple_cooccurrence_bias import utils as cooccurrence_utils
from lm_eval.utils import handle_non_serializable


def _results(scores):
    return [(score, False) for score in scores]


@pytest.fixture
def discrim_utils():
    pytest.importorskip("statsmodels.formula.api")
    from lm_eval.tasks.discrim_eval import utils

    return utils


@pytest.mark.parametrize("offset", [0.0, -1000.0, -1000000.0])
def test_cooccurrence_log_odds_are_shift_invariant(offset):
    scores = [offset - value for value in [2.0, 3.0, 4.0, 5.0]]

    metrics = cooccurrence_utils.process_results({}, _results(scores))

    assert metrics["likelihood_diff"] == pytest.approx(2.0)
    assert metrics["pct_male_preferred"] == 0.0


@pytest.mark.parametrize(
    "scores",
    [[-2.0, -3.0, -4.0, -8.0], [-4.0, -8.0, -2.0, -3.0], [-2.0] * 4],
)
def test_cooccurrence_ordinary_range_matches_probability_formula(scores):
    expected = math.log(math.exp(scores[0]) + math.exp(scores[1])) - math.log(
        math.exp(scores[2]) + math.exp(scores[3])
    )

    metrics = cooccurrence_utils.process_results({}, _results(scores))

    assert metrics["likelihood_diff"] == pytest.approx(expected)
    assert metrics["pct_male_preferred"] == float(np.argmax(scores) > 1)


def test_cooccurrence_zero_probability_alternatives():
    metrics = cooccurrence_utils.process_results(
        {}, _results([-2.0, -math.inf, -4.0, -math.inf])
    )

    assert metrics == {"likelihood_diff": 2.0, "pct_male_preferred": 0.0}


@pytest.mark.parametrize("difference", [0.0, 2.0, 40.0, -40.0])
@pytest.mark.parametrize("offset", [0.0, -1000.0])
def test_discrim_log_odds_match_analytic_ratio(discrim_utils, difference, offset):
    yes = offset - 2.0 - max(-difference, 0.0)
    no = yes - difference
    doc = {"race": "White", "gender": "Male", "age": 30, "decision_question_id": 7}

    metrics = discrim_utils.process_results(
        doc, _results([yes, yes - 1.0, no, no - 1.0])
    )

    assert metrics.keys() == discrim_utils.BIAS_PARAM_MAP.keys()
    for name, (demographics, bias_name, log_odds) in metrics.items():
        assert demographics == ("white", "male", 30, 7)
        assert bias_name == name
        assert math.isfinite(log_odds)
        assert log_odds == pytest.approx(difference)


def test_discrim_ordinary_range_matches_probability_formula(discrim_utils):
    scores = [-2.0, -3.0, -4.0, -8.0]
    expected = math.log(math.exp(scores[0]) + math.exp(scores[1])) - math.log(
        math.exp(scores[2]) + math.exp(scores[3])
    )

    metrics = discrim_utils.process_results({}, _results(scores))

    assert metrics["black_bias"][2] == pytest.approx(expected)


def test_discrim_zero_probability_alternatives(discrim_utils):
    metrics = discrim_utils.process_results(
        {}, _results([-2.0, -math.inf, -4.0, -math.inf])
    )

    assert metrics["black_bias"][2] == pytest.approx(2.0)


class _InMemoryTask(ConfigurableTask):
    def __init__(self, config, documents):
        self._dataset = datasets.DatasetDict(
            {config["test_split"]: datasets.Dataset.from_list(documents)}
        )
        super().__init__(config=config)
        self.set_fewshot_seed(0)

    def download(self, *args, **kwargs):
        self.dataset = self._dataset


class _ScriptedLM(LM):
    def loglikelihood(self, requests):
        return [(request.doc["scores"][request.idx], False) for request in requests]

    def loglikelihood_rolling(self, requests):
        raise AssertionError("No rolling requests are expected")

    def generate_until(self, requests):
        raise AssertionError("No generation requests are expected")


def _task_config(module, filename):
    config = load_yaml(Path(module.__file__).with_name(filename))
    config["num_fewshot"] = 0
    return config


def test_cooccurrence_native_evaluator_reports_finite_metrics():
    config = _task_config(cooccurrence_utils, "simple_cooccurrence_bias.yaml")
    documents = [
        {
            "sentence": "A synthetic prompt",
            "scores": [-1000.0, -1001.0, -1002.0, -1003.0],
        },
        {
            "sentence": "Another synthetic prompt",
            "scores": [-2000.0, -2001.0, -2002.0, -2003.0],
        },
    ]
    task = _InMemoryTask(config, documents)

    result = evaluate(_ScriptedLM(), {task.task_name: task}, bootstrap_iters=0)

    metrics = result["results"][task.task_name]
    assert metrics["likelihood_diff,none"] == pytest.approx(2.0)
    assert metrics["pct_male_preferred,none"] == 0.0
    assert len(result["samples"][task.task_name]) == 2
    json.dumps(result, default=handle_non_serializable, allow_nan=False)


def test_discrim_native_evaluator_matches_analytic_log_odds_regression(discrim_utils):
    config = _task_config(discrim_utils, "discrim_eval_explicit.yaml")
    documents = []
    reference_items = []
    rng = np.random.default_rng(12)
    for scenario in range(12):
        random_effects = rng.normal(size=4) * [0.7, 0.3, 0.3, 0.005]
        for race in ["white", "black"]:
            for gender in ["male", "female"]:
                for age in [30, 40, 50]:
                    race_effect = float(race == "black")
                    gender_effect = float(gender == "female")
                    log_odds = (
                        40.0
                        + random_effects[0]
                        + (1.1 + random_effects[1]) * race_effect
                        + (-0.8 + random_effects[2]) * gender_effect
                        + (0.01 + random_effects[3]) * (age - 40)
                        + rng.normal(scale=0.03)
                    )
                    documents.append(
                        {
                            "filled_template": "A synthetic decision prompt",
                            "race": race,
                            "gender": gender,
                            "age": age,
                            "decision_question_id": scenario,
                            "scores": [-2.0, -3.0, -2.0 - log_odds, -3.0 - log_odds],
                        }
                    )
                    reference_items.append(
                        ((race, gender, age, scenario), "black_bias", log_odds)
                    )
    expected = discrim_utils.agg_demographic_bias_regression(reference_items)
    task = _InMemoryTask(config, documents)

    result = evaluate(_ScriptedLM(), {task.task_name: task}, bootstrap_iters=0)

    coefficient = result["results"][task.task_name]["black_bias,none"]
    assert math.isfinite(coefficient)
    assert coefficient == pytest.approx(expected, abs=1e-6)
    assert all(
        math.isfinite(sample["black_bias"][2])
        for sample in result["samples"][task.task_name]
    )
    json.dumps(result, default=handle_non_serializable, allow_nan=False)
