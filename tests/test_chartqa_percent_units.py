from pathlib import Path

import datasets
import pytest

from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks._yaml_loader import load_yaml
from lm_eval.tasks.chartqa import utils


@pytest.mark.parametrize(
    "prediction,target,expected",
    [
        ("1%", "100%", 0.0),
        ("100%", "1%", 0.0),
        ("0.5%", "50%", 0.0),
        ("50%", "0.5%", 0.0),
        ("-1%", "-100%", 0.0),
        ("-100%", "-1%", 0.0),
        ("25%", "25%", 1.0),
        ("104%", "100%", 1.0),
        ("106%", "100%", 0.0),
        ("0%", "0%", 1.0),
        ("25%", "0.25", 1.0),
        ("0.25", "25%", 1.0),
        ("25%", "25", 1.0),
        ("25", "25%", 1.0),
        ("104", "100", 1.0),
        ("106", "100", 0.0),
    ],
)
@pytest.mark.parametrize("mode", ["raw", "explicit", "anywhere"])
def test_explicit_percentages_share_units_without_one_sided_rescaling(
    prediction, target, expected, mode
):
    if mode == "raw":
        actual = utils.RelaxedCorrectness().score(prediction, target)
    elif mode == "explicit":
        actual = utils.relaxed_accuracy([target], [f"Final Answer: {prediction}"])[
            "relaxed_accuracy"
        ]
    else:
        actual = utils.anywhere_accuracy([target], [f"Final Answer: {prediction}"])[
            "anywhere_accuracy"
        ]

    assert actual == expected


def test_multiple_targets_still_accept_a_matching_percentage():
    scorer = utils.RelaxedCorrectness()

    assert scorer.score("1%", ["100%", "1%"]) == 1.0
    assert scorer.score("1%", ["100%", "50%"]) == 0.0


@pytest.mark.parametrize(
    "prediction,target,expected",
    [
        ("There are 1,000 items.", "1000", 1.0),
        ("There are 1000 items.", "1000", 1.0),
        ("This is blue.", "blue", 1.0),
        ("No match.", "blue", 0.0),
    ],
)
def test_anywhere_fallback_without_an_explicit_final_answer_is_unchanged(
    prediction, target, expected
):
    assert (
        utils.AnywhereInAnswerRelaxedCorrectness().score(prediction, target) == expected
    )


@pytest.mark.parametrize("task_name", ["chartqa", "chartqa_llama", "chartqa_llama_90"])
@pytest.mark.parametrize("prediction,expected", [("1%", 0.0), ("104%", 1.0)])
def test_real_chartqa_configurations_use_consistent_percentage_units(
    task_name, prediction, expected
):
    doc = {"query": "What percentage is shown?", "label": ["100%"], "image": None}

    class LocalChartQA(ConfigurableTask):
        def download(self, dataset_kwargs=None, **kwargs):
            self.dataset = datasets.DatasetDict(test=datasets.Dataset.from_list([doc]))

    task = LocalChartQA(
        config=load_yaml(
            Path(utils.__file__).parent / f"{task_name}.yaml", resolve_func=True
        )
    )
    request = task.construct_requests(
        doc, task.doc_to_text(doc), metadata=(task.config.task, 0, 1)
    )

    assert request.request_type == "generate_until"
    assert task.process_results(doc, [f"Final Answer: {prediction}"]) == {
        "exact_match": 0.0,
        "relaxed_accuracy": expected,
        "anywhere_accuracy": expected,
    }
