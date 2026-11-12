from pathlib import Path

import datasets
import pytest

from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks._yaml_loader import load_yaml
from lm_eval.tasks.mmmu import utils


@pytest.mark.parametrize(
    "text,expected",
    [
        ("The answer is 1.25e3.", ["1.25e3"]),
        ("The answer is -1.25e-3.", ["-1.25e-3"]),
        ("The answer is 1,234.5.", ["1,234.5"]),
        ("The answer is -1,234.5e-2.", ["-1,234.5e-2"]),
        ("The answer is .5e2.", [".5e2"]),
        ("The answer is -.5e-2.", ["-.5e-2"]),
        ("The answer is 12e+3.", ["12e+3"]),
        ("The answer is -2.5.", ["-2.5"]),
        ("The answer is .75.", [".75"]),
        ("The answer is 1,234,567.", ["1,234,567"]),
        ("The answer is 7.", ["7"]),
        ("No numerical answer.", []),
    ],
)
def test_extract_numbers_keeps_complete_nonoverlapping_candidates(text, expected):
    assert utils.extract_numbers(text) == expected


@pytest.mark.parametrize(
    "response,gold,expected",
    [
        ("The answer is 1.25e3.", "3", 0.0),
        ("The answer is 1.25e3.", "1", 0.0),
        ("The answer is 1.25e3.", "1250", 1.0),
        ("The answer is -1.25e-3.", "-3", 0.0),
        ("The answer is 1,234.5.", "1234", 0.0),
        ("The answer is 1,234.5.", "234.5", 0.0),
        ("The answer is 1,234.5.", "1234.5", 1.0),
        ("The answer is .5e2.", "50", 1.0),
        ("The answer is .5e2.", "500", 0.0),
        ("The answer is 1.25e3 and 7.", "7", 1.0),
    ],
)
def test_mmmu_does_not_score_numeric_fragments_as_separate_answers(
    response, gold, expected
):
    doc = {"question_type": "open", "answer": gold}

    assert utils.process_results(doc, [response]) == {"acc": expected}


@pytest.mark.parametrize("answer", ["A", "B", "C"])
def test_numeric_extraction_change_keeps_multiple_choice_scoring(answer):
    doc = {
        "question_type": "multiple-choice",
        "options": "['one', 'two', 'three']",
        "answer": answer,
    }

    assert utils.process_results(doc, [f"({answer})"]) == {"acc": 1.0}


@pytest.mark.parametrize("gold,expected", [("3", 0.0), ("1250", 1.0)])
def test_mmmu_math_configuration_scores_complete_numbers(gold, expected):
    doc = {
        "question": "Give the result.",
        "question_type": "open",
        "answer": gold,
        "options": "[]",
    }

    class LocalMMMU(ConfigurableTask):
        def download(self, dataset_kwargs=None, **kwargs):
            self.dataset = datasets.DatasetDict(
                validation=datasets.Dataset.from_list([doc])
            )

    task = LocalMMMU(
        config=load_yaml(
            Path(utils.__file__).parent / "mmmu_math.yaml", resolve_func=True
        )
    )
    request = task.construct_requests(
        doc, task.doc_to_text(doc), metadata=(task.config.task, 0, 1)
    )

    assert request.request_type == "generate_until"
    assert task.process_results(doc, ["The answer is 1.25e3."]) == {"acc": expected}
