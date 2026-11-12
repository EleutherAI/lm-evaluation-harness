"""Exact SCORE consistency results across response counts and task entry points."""

import copy
import json
import random
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import datasets
import pytest

from lm_eval.api.task import ConfigurableTask
from lm_eval.evaluator_utils import _compute_task_aggregations
from lm_eval.tasks._yaml_loader import load_yaml
from lm_eval.tasks.score import non_greedy_summarizer, utils
from lm_eval.tasks.score.math import utils_math


def pairwise_reference(responses):
    matches = pairs = 0
    for answers in responses:
        for left, right in combinations(answers, 2):
            matches += left == right
            pairs += 1
    return matches / pairs if pairs else 0.0


@pytest.mark.parametrize(
    "function",
    [
        utils.calculate_consistency_rate,
        non_greedy_summarizer.calculate_consistency_rate,
    ],
)
@pytest.mark.parametrize(
    "responses",
    [
        [],
        [[], ["A"], []],
        [["A"] * 10],
        [["A", "B", "C", "D", "E"]],
        [["A", "A", "B", "B", "A"]],
        [["A", "A"], ["B", "C", "D", "E"]],
        [["A", "A", "B"], ["A", "B", "B", "C", "C"], []],
        [["", "", "A", "a", " A", "A "]],
        [["猫", "猫", "犬", "é", "e\u0301", "é"]],
        [["A"] * 513 + ["B"] * 257 + ["C"] * 130],
    ],
)
def test_consistency_matches_pairwise_definition(function, responses):
    original = copy.deepcopy(responses)
    assert function(responses) == pairwise_reference(responses)
    assert responses == original


@pytest.mark.parametrize(
    "function",
    [
        utils.calculate_consistency_rate,
        non_greedy_summarizer.calculate_consistency_rate,
    ],
)
def test_randomized_ragged_groups_keep_pair_weighting(function):
    rng = random.Random(287)
    responses = [rng.choices(["A", "B", "C", "", "犬"], k=n) for n in range(31)]
    expected = pairwise_reference(responses)
    assert function(responses) == expected
    for row in responses:
        rng.shuffle(row)
    rng.shuffle(responses)
    assert function(responses) == expected


def test_native_prompt_task_keeps_consistency_and_accuracy():
    task_path = (
        Path(__file__).parents[1]
        / "lm_eval/tasks/score/mmlu_pro/score_prompt_robustness_mmlu_pro.yaml"
    )
    config = load_yaml(task_path)
    data = datasets.DatasetDict(
        {
            "test": datasets.Dataset.from_list(
                [
                    {
                        "question_id": index,
                        "question": f"Question {index}",
                        "options": ["first", "second", "third", "fourth"],
                        "answer_index": 0,
                        "answer": "A",
                        "category": "test",
                    }
                    for index in range(2)
                ]
            )
        }
    )
    config.update(custom_dataset=lambda **kwargs: data, validation_split=None)
    task = ConfigurableTask(config=config)
    by_question = [["A"] * 6 + ["B"] * 4, ["A", "B", "C", "D"] * 2 + ["A", "B"]]
    raw_metrics = defaultdict(list)
    assert len(task.eval_docs) == 20
    for doc in task.eval_docs:
        response = by_question[doc["question_id"]][doc["prompt_id"]]
        for metric, value in task.process_results(doc, [response]).items():
            raw_metrics[metric, "none"].append(value)

    aggregate, count = _compute_task_aggregations(task, raw_metrics, 0)
    assert count == 20
    assert aggregate["consistency_rate,none"] == pairwise_reference(by_question)
    for prompt in range(10):
        expected = sum(row[prompt] == "A" for row in by_question) / 2
        assert aggregate[f"{prompt}_macro_accuracy,none"] == expected


def test_non_greedy_summary_keeps_persisted_consistency(tmp_path, monkeypatch):
    responses = [["A", "A", "B", "B", "A"], ["C"] * 5]
    for seed in range(1, 6):
        directory = tmp_path / f"seed_{seed}" / "model"
        directory.mkdir(parents=True)
        records = [
            {"non_greedy_macro_accuracy": [index, row[seed - 1], "A", "test"]}
            for index, row in enumerate(responses)
        ]
        path = directory / "samples_score_non_greedy_robustness_mmlu_pro_test.jsonl"
        path.write_text("\n".join(json.dumps(record) for record in records))
    monkeypatch.setattr(
        sys, "argv", ["summarizer", "--log_dir", str(tmp_path), "--dataset", "mmlu_pro"]
    )
    non_greedy_summarizer.main()
    [output] = tmp_path.glob("mmlu_pro_results_*.json")
    metrics = json.loads(output.read_text())["results"][
        "score_non_greedy_robustness_mmlu_pro"
    ]
    assert metrics["consistency_rate,none"] == pairwise_reference(responses)
    for seed in range(1, 6):
        expected = sum(row[seed - 1] == "A" for row in responses) / 2
        assert metrics[f"seed_{seed}_accuracy,none"] == expected


@pytest.mark.parametrize(
    "function",
    [
        utils_math.calculate_consistency_rate,
        non_greedy_summarizer.calculate_math_consistency_rate,
    ],
)
def test_math_consistency_still_uses_numeric_equivalence(function):
    assert function([["0.5", "0.50", "2"], ["3.0", "3"]]) == 0.5
