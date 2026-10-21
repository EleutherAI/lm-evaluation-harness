import random
from importlib import import_module

import datasets
import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "lm_eval.tasks.gpqa.zeroshot.utils",
        "lm_eval.tasks.gpqa.n_shot.utils",
        "lm_eval.tasks.gpqa.generative.utils",
        "lm_eval.tasks.gpqa.cot_zeroshot.utils",
        "lm_eval.tasks.gpqa.cot_n_shot.utils",
        "lm_eval.tasks.leaderboard.gpqa.utils",
    ],
)
def test_process_docs_is_independent_of_global_random_state(module_name):
    utils = import_module(module_name)
    dataset = datasets.Dataset.from_dict(
        {
            "Question": ["Which answer is correct?"],
            "Incorrect Answer 1": ["First incorrect answer"],
            "Incorrect Answer 2": ["Second incorrect answer"],
            "Incorrect Answer 3": ["Third incorrect answer"],
            "Correct Answer": ["Correct answer"],
        }
    )

    random_state = random.getstate()
    try:
        random.seed(1)
        first = utils.process_docs(dataset)[0]
        random.seed(1234)
        second = utils.process_docs(dataset)[0]
    finally:
        random.setstate(random_state)

    choice_columns = ["choice1", "choice2", "choice3", "choice4", "answer"]
    assert {column: first[column] for column in choice_columns} == {
        column: second[column] for column in choice_columns
    }
