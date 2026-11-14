from pathlib import Path

import datasets
import pytest

from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks._yaml_loader import load_yaml
from lm_eval.tasks.mutual import utils


@pytest.mark.parametrize("gold,rank", [("A", 4), ("B", 1), ("C", 3), ("D", 2)])
@pytest.mark.parametrize(
    "greedy", [(False, False, False, False), (False, True, True, False)]
)
def test_mutual_ranks_loglikelihoods_without_greedy_flags(gold, rank, greedy):
    results = list(zip([-4.0, -1.0, -3.0, -2.0], greedy, strict=True))

    scores = utils.process_results({"answers": gold}, results)

    assert scores == {"r@1": int(rank == 1), "r@2": int(rank <= 2), "mrr": 1 / rank}


@pytest.mark.parametrize("gold,rank", [("A", 1), ("B", 2), ("C", 3), ("D", 4)])
def test_equal_loglikelihoods_use_choice_order_consistently(gold, rank):
    results = [(-1.0, False), (-1.0, True), (-1.0, False), (-1.0, True)]

    scores = utils.process_results({"answers": gold}, results)

    assert scores == {"r@1": int(rank == 1), "r@2": int(rank <= 2), "mrr": 1 / rank}


@pytest.mark.parametrize("task_file", ["mutual.yaml", "multual_plus.yaml"])
def test_mutual_task_configuration_receives_loglikelihood_pairs(task_file):
    doc = {
        "article": "Choose the answer.",
        "options": ["first", "second", "third", "fourth"],
        "answers": "C",
    }

    class LocalMutualTask(ConfigurableTask):
        def download(self, dataset_kwargs=None, **kwargs):
            self.dataset = datasets.DatasetDict(
                train=datasets.Dataset.from_list([doc]),
                validation=datasets.Dataset.from_list([doc]),
            )

    config_path = Path(utils.__file__).parent / task_file
    task = LocalMutualTask(config=load_yaml(config_path, resolve_func=True))
    requests = task.construct_requests(
        doc, task.doc_to_text(doc), metadata=(task.config.task, 0, 1)
    )
    assert [request.request_type for request in requests] == ["loglikelihood"] * 4

    scores = task.process_results(
        doc, [(-4.0, False), (-3.0, False), (-1.0, False), (-2.0, False)]
    )

    assert scores == {"r@1": 1, "r@2": 1, "mrr": 1.0}
