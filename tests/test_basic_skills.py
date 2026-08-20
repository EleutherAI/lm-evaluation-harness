from __future__ import annotations

import random
from pathlib import Path

import pytest

from lm_eval.tasks._yaml_loader import load_yaml
from lm_eval.tasks.basic_skills import utils


FIXTURES = {
    "arithmetic": {
        "id": "0",
        "question": "4 * 8 =",
        "answer": "32",
        "wrong_answers": ["28", "30", "34", "36", "40", "42", "16", "12", "24"],
        "choices": ["40", "24", "36", "32", "28", "12", "16", "34", "30", "42"],
        "gold": 3,
    },
    "coding": {
        "id": "0",
        "question": 'def add(a, b):\n    """Return the sum of a and b."""\n    return',
        "answer": "a + b",
        "wrong_answers": ["a - b", "print(a + b)", "(a, b)", "a * b"],
        "choices": ["a * b", "print(a + b)", "a + b", "a - b", "(a, b)"],
        "gold": 2,
    },
    "common_knowledge": {
        "id": "0",
        "question": "The number of legs of a cat is",
        "answer": "four",
        "wrong_answers": ["two", "three", "five"],
        "choices": ["five", "two", "four", "three"],
        "gold": 2,
    },
    "logical_reasoning": {
        "id": "0",
        "question": "All mammals are warm-blooded. A whale is a mammal.",
        "answer": "Hence, a whale is warm-blooded.",
        "wrong_answers": [
            "Hence, a whale is cold-blooded.",
            "Hence, a whale is not an animal.",
            "Hence, some mammals are cold-blooded.",
        ],
        "choices": [
            "Hence, some mammals are cold-blooded.",
            "Hence, a whale is cold-blooded.",
            "Hence, a whale is warm-blooded.",
            "Hence, a whale is not an animal.",
        ],
        "gold": 2,
    },
    "string_operations": {
        "id": "0",
        "question": "The length of the string 'hello' is",
        "answer": "5",
        "wrong_answers": ["4", "6", "7", "0"],
        "choices": ["0", "6", "5", "4", "7"],
        "gold": 2,
    },
    "pattern": {
        "id": "0",
        "question": "2 4 6 8",
        "answer": "10",
        "wrong_answers": ["9", "12", "14"],
        "choices": ["14", "9", "10", "12"],
        "gold": 2,
    },
}


@pytest.mark.parametrize("domain", FIXTURES)
def test_process_docs_matches_pinned_olmes(domain):
    source = FIXTURES[domain]
    raw_doc = {
        key: source[key] for key in ("id", "question", "answer", "wrong_answers")
    }
    original_wrong_answers = raw_doc["wrong_answers"].copy()

    rc = utils._process_rc_doc(raw_doc)
    mc = utils._process_mc_doc(raw_doc)

    assert raw_doc["wrong_answers"] == original_wrong_answers
    assert rc == {
        "question": source["question"],
        "choices": source["choices"],
        "gold": source["gold"],
    }
    assert mc["gold"] == source["gold"]
    assert mc["choices"] == list(utils.CHOICE_LABELS[: len(source["choices"])])
    assert mc["question"] == utils.make_mcq_prompt(
        source["question"], source["choices"]
    )


def test_mc_prompt_snapshot():
    assert utils.make_mcq_prompt("2 4 6 8", ["14", "9", "10", "12"]) == (
        "2 4 6 8\n A. 14\n B. 9\n C. 10\n D. 12\nAnswer:"
    )


def test_fewshot_sampler_matches_stateless_olmes_sampling():
    pool = [{"id": str(i)} for i in range(10)]
    seed = 1234
    sampled_with_extra = random.Random(seed).sample(pool, 6)
    eval_doc = sampled_with_extra[2]
    expected = [doc for doc in sampled_with_extra if doc != eval_doc][:5]

    sampler = utils.OLMESContextSampler(pool, rnd=seed)

    assert sampler.sample(5, eval_doc=eval_doc) == expected
    assert sampler.sample(5, eval_doc=eval_doc) == expected


@pytest.mark.parametrize("prompt_form", ["rc", "mc"])
@pytest.mark.parametrize("domain", FIXTURES)
def test_task_configs_are_pinned_and_report_olmes_metric(domain, prompt_form):
    task_dir = Path(utils.__file__).parent
    config = load_yaml(task_dir / f"{domain}_{prompt_form}.yaml")

    assert config["task"] == f"basic_skills_{domain}_{prompt_form}"
    assert config["dataset_path"] == "json"
    assert config["dataset_name"] == domain
    revision = "faf63e9719e124d7741519a024719c8992622630"
    assert config["metadata"]["dataset_repository"] == "allenai/basic-skills"
    assert config["metadata"]["dataset_revision"] == revision
    assert config["dataset_kwargs"]["data_files"]["validation"] == (
        "https://huggingface.co/datasets/allenai/basic-skills/resolve/"
        f"{revision}/{domain}/validation.json"
    )
    assert config["num_fewshot"] == 5
    assert config["test_split"] == config["fewshot_split"] == "validation"
    assert config["tag"] == ["multiple_choice", "olmes_basic_skills"]
    assert [metric["metric"] for metric in config["metric_list"]] == [
        "acc_per_token",
        "acc",
        "acc_norm",
    ]
