import copy
import random
import re
from pathlib import Path

import datasets
import pytest

from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks._yaml_loader import load_yaml
from lm_eval.tasks.med_prescriptions import utils


@pytest.fixture(autouse=True)
def preserve_random_state():
    state = random.getstate()
    random.seed(17)
    yield
    random.setstate(state)


def prescription_dataset(medicine_count):
    record = {
        "procedure": {"chief_complaints_diagnosis": "dataset fixture"},
        "medicine_details": [
            {
                "medicine_name": f"Medicine {index}",
                "medicine_frequency": "once daily",
                "course_duration": f"{index + 1} days",
            }
            for index in range(medicine_count)
        ],
    }
    return datasets.Dataset.from_list([{"results": repr(record)}])


def prompt_choices(prompt):
    return re.findall(r"^[A-D]\. (.*?)\s*$", prompt, re.MULTILINE)


@pytest.mark.parametrize("difficulty", ["easy", "hard"])
@pytest.mark.parametrize("medicine_count", [1, 2, 3, 4])
def test_prompt_and_scored_choices_stay_identical(difficulty, medicine_count):
    doc = utils.process_docs(prescription_dataset(medicine_count))[0]
    original = copy.deepcopy(doc)
    to_choice = getattr(utils, f"doc_to_choice_{difficulty}")
    to_text = getattr(utils, f"doc_to_text_{difficulty}")

    choices = to_choice(doc)
    prompt = to_text(doc)
    assert prompt_choices(prompt) == choices
    assert len(choices) == 4
    assert choices[utils.doc_to_target(doc)] == doc["gold"]

    # Unrelated sampling must not change the candidates of an existing question.
    random.random()
    state = random.getstate()
    assert to_choice(doc) == choices
    assert to_text(doc) == prompt
    assert random.getstate() == state
    assert doc == original


@pytest.mark.parametrize("difficulty", ["easy", "hard"])
def test_native_task_requests_score_the_displayed_options(difficulty):
    path = Path(utils.__file__).parent / f"med_prescriptions_{difficulty}.yaml"
    config = load_yaml(path, resolve_func=True)
    raw = prescription_dataset(4)
    config["custom_dataset"] = lambda **kwargs: datasets.DatasetDict({"train": raw})
    config["num_fewshot"] = 0
    task = ConfigurableTask(config=config)
    doc = task.task_docs[0]
    prompt = task.doc_to_text(doc)

    requests = task.construct_requests(doc, prompt)
    displayed = prompt_choices(prompt)
    assert len(requests) == len(displayed) == 4
    assert [request.arguments for request in requests] == [
        (prompt, task.config.target_delimiter + option) for option in displayed
    ]
    assert displayed[task.doc_to_target(doc)] == doc["gold"]
    scores = [(-10.0, False) for _ in displayed]
    scores[task.doc_to_target(doc)] = (-1.0, False)
    assert task.process_results(doc, scores)["acc"] == 1.0


@pytest.mark.parametrize("difficulty", ["easy", "hard"])
def test_processed_choices_survive_dataset_serialization(tmp_path, difficulty):
    processed = utils.process_docs(prescription_dataset(3))
    to_choice = getattr(utils, f"doc_to_choice_{difficulty}")
    to_text = getattr(utils, f"doc_to_text_{difficulty}")
    expected_choices = to_choice(processed[0])
    expected_prompt = to_text(processed[0])
    processed.save_to_disk(tmp_path / "prescriptions")
    restored = datasets.load_from_disk(tmp_path / "prescriptions")

    random.seed(999)
    assert to_choice(restored[0]) == expected_choices
    assert to_text(restored[0]) == expected_prompt


def test_preprocessing_preserves_record_filtering_and_gold():
    valid = prescription_dataset(2)
    empty = prescription_dataset(0)
    processed = utils.process_docs(datasets.concatenate_datasets([empty, valid]))

    assert len(processed) == 1
    doc = processed[0]
    assert doc["gold"] in utils.get_medicines_list(valid[0])
    for to_choice in (utils.doc_to_choice_easy, utils.doc_to_choice_hard):
        choices = to_choice(doc)
        assert len(choices) == 4
        assert choices[doc["gold_position"]] == doc["gold"]
