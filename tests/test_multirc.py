from pathlib import Path

import datasets
import pytest

from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks._yaml_loader import load_yaml


MULTIRC_CONFIG = (
    Path(__file__).parents[1]
    / "lm_eval"
    / "tasks"
    / "super_glue"
    / "multirc"
    / "default.yaml"
)

# SuperGLUE MultiRC: label is ClassLabel(names=["False", "True"]), i.e. 1 means the
# candidate answer is correct for the question.
DOCS = [
    {
        "paragraph": "The sky is blue.",
        "question": "What colour is the sky?",
        "answer": "Blue",
        "idx": {"paragraph": 0, "question": 0, "answer": 0},
        "label": 1,
    },
    {
        "paragraph": "The sky is blue.",
        "question": "What colour is the sky?",
        "answer": "Green",
        "idx": {"paragraph": 0, "question": 0, "answer": 1},
        "label": 0,
    },
]


@pytest.fixture(scope="module")
def multirc_task():
    config = load_yaml(MULTIRC_CONFIG, resolve_func=False)
    config.pop("tag", None)
    config["custom_dataset"] = lambda **kwargs: datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_list(DOCS),
            "validation": datasets.Dataset.from_list(DOCS),
        }
    )
    return ConfigurableTask(config=config)


@pytest.mark.parametrize("doc", DOCS, ids=["correct_answer", "wrong_answer"])
def test_multirc_gold_choice_matches_label(multirc_task, doc):
    gold = multirc_task.doc_to_choice(doc)[multirc_task.doc_to_target(doc)]
    expected = "yes" if doc["label"] == 1 else "no"
    assert gold.endswith(f"Is the answer correct? {expected}")


@pytest.mark.parametrize("doc", DOCS, ids=["correct_answer", "wrong_answer"])
def test_multirc_oracle_scores_full_accuracy(multirc_task, doc):
    choices = multirc_task.doc_to_choice(doc)
    # A model that puts more likelihood on the true yes/no judgement must score 1.
    truth = "yes" if doc["label"] == 1 else "no"
    results = [(-1.0 if c.endswith(truth) else -5.0, False) for c in choices]
    assert multirc_task.process_results(doc, results)["acc"] == 1.0
