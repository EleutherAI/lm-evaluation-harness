"""Class-level task presets must not share runtime changes between instances."""

import pytest
from datasets import Dataset, DatasetDict

from lm_eval.api.task import ConfigurableTask
from lm_eval.config.task import TaskConfig
from lm_eval.tasks._factory import TaskFactory
from lm_eval.tasks._index import Entry, Kind


def local_dataset(**kwargs):
    return DatasetDict(
        {
            "test": Dataset.from_list(
                [{"question": "Question?", "answer": "yes", "alternative": "no"}]
            )
        }
    )


@pytest.fixture
def preset_task():
    class PresetTask(ConfigurableTask):
        CONFIG = TaskConfig(
            task="preset",
            custom_dataset=local_dataset,
            test_split="test",
            doc_to_text="question",
            doc_to_target="answer",
            num_fewshot=0,
            generation_kwargs={"until": ["END"], "temperature": 0.0},
        )

    return PresetTask


def test_later_instance_does_not_change_existing_task_scoring(preset_task):
    first = preset_task(config={"task": "first"})
    doc = first.eval_docs[0]
    assert first.process_results(doc, ["yes"])["exact_match"] == 1.0

    second = preset_task(config={"task": "second", "doc_to_target": "alternative"})

    assert first.process_results(doc, ["yes"])["exact_match"] == 1.0
    assert second.process_results(doc, ["yes"])["exact_match"] == 0.0
    assert first.task_name == "first"
    assert second.task_name == "second"
    assert preset_task.CONFIG.task == "preset"
    assert preset_task.CONFIG.doc_to_target == "answer"


def test_runtime_generation_changes_do_not_escape_instance(preset_task):
    first = preset_task()
    second = preset_task()
    first.set_config("generation_kwargs", {"temperature": 0.7}, update=True)
    first.config.generation_kwargs["until"].append("STOP")

    doc = second.eval_docs[0]
    request = second.construct_requests(doc, "Question?")
    assert request.args[1] == {"until": ["END"], "temperature": 0.0}
    assert preset_task().construct_requests(doc, "Question?").args[1] == {
        "until": ["END"],
        "temperature": 0.0,
    }


def test_factory_keeps_class_task_names_independent(preset_task):
    factory = TaskFactory()
    entries = {
        name: Entry(
            name=name,
            kind=Kind.PY_TASK,
            yaml_path=None,
            cfg={"task": name, "class": preset_task},
        )
        for name in ["first", "second"]
    }
    first = factory.build(entries["first"], registry=entries)
    second = factory.build(entries["second"], registry=entries)

    assert first.task_name == "first"
    assert second.task_name == "second"
    assert first.dump_config()["task"] == "first"
    assert preset_task.CONFIG.task == "preset"
