import copy
import json

import pytest
import yaml

from lm_eval.api.task import ConfigurableTask
from lm_eval.config.evaluate_config import EvaluatorConfig
from lm_eval.tasks import TaskManager


class LocalPythonTask(ConfigurableTask):
    def __init__(self, config):
        super().__init__(config={k: v for k, v in config.items() if k != "class"})


@pytest.fixture
def local_tasks(tmp_path):
    data_path = tmp_path / "data.json"
    data_path.write_text(
        json.dumps(
            [
                {"question": "2+2?", "choices": ["3", "4"], "answer": 1},
                {"question": "3+3?", "choices": ["6", "7"], "answer": 0},
            ]
        ),
        encoding="utf-8",
    )
    configs = []
    for name in ["config_leaf_a", "config_leaf_b"]:
        config = {
            "task": name,
            "tag": "config_tag",
            "dataset_path": "json",
            "dataset_kwargs": {"data_files": {"test": str(data_path)}},
            "test_split": "test",
            "output_type": "multiple_choice",
            "doc_to_text": "question",
            "doc_to_choice": "choices",
            "doc_to_target": "answer",
            "num_fewshot": 0,
            "metric_list": [{"metric": "acc", "aggregation": "mean"}],
            "metadata": {"leaf": name},
        }
        (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        configs.append(config)
    inner = {"group": "config_inner", "task": [c["task"] for c in configs]}
    (tmp_path / "inner.yaml").write_text(yaml.safe_dump(inner), encoding="utf-8")
    return tmp_path, configs


@pytest.mark.parametrize(
    ("members", "expected_fewshot"),
    [
        (["config_leaf_a", "config_leaf_b"], 1),
        ([{"group": "config_inner", "num_fewshot": 2}], 2),
        (["config_tag"], 1),
        (
            [
                {
                    "group": "inline",
                    "task": ["config_leaf_a", "config_leaf_b"],
                }
            ],
            1,
        ),
    ],
)
def test_group_dict_matches_registered_group(local_tasks, members, expected_fewshot):
    directory, _ = local_tasks
    config = {
        "group": "config_group",
        "group_alias": "Readable group",
        "task": members,
        "num_fewshot": 1,
        "aggregate_metric_list": [{"metric": "acc", "aggregation": "mean"}],
        "metadata": {"group_only": "preserved"},
    }
    original = copy.deepcopy(config)
    (directory / "group.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    manager = TaskManager(
        include_path=directory, include_defaults=False, metadata={"runtime": True}
    )

    by_name = manager.load("config_group")
    by_dict = manager.load(config)

    assert config == original
    assert by_dict["group_map"] == by_name["group_map"]
    assert set(by_dict["tasks"]) == {"config_leaf_a", "config_leaf_b"}
    group = by_dict["groups"]["config_group"]
    assert group.alias == "Readable group"
    assert (
        group.aggregate_metric_list
        == by_name["groups"]["config_group"].aggregate_metric_list
    )
    assert group.metadata["group_only"] == "preserved"
    assert group.metadata["runtime"] is True
    for name, task in by_dict["tasks"].items():
        assert task.config.num_fewshot == expected_fewshot
        assert task.config.num_fewshot == by_name["tasks"][name].config.num_fewshot
        assert task.config.metadata["leaf"] == name
        assert task.config.metadata["runtime"] is True
        assert "group_only" not in task.config.metadata
        assert len(task.test_docs()) == 2


@pytest.mark.parametrize("reference_kind", ["path", "glob", "directory"])
def test_cli_preloaded_group_dict_keeps_group_fields(
    local_tasks, monkeypatch, reference_kind
):
    directory, configs = local_tasks
    suite_dir = directory / "suite"
    suite_dir.mkdir()
    group_path = suite_dir / "group.yaml"
    group_path.write_text(
        yaml.safe_dump(
            {
                "group": "inline_suite",
                "group_alias": "Inline suite",
                "aggregate_metric_list": [{"metric": "acc"}],
                "task": configs,
            }
        ),
        encoding="utf-8",
    )
    reference = {
        "path": group_path,
        "glob": suite_dir / "*.yaml",
        "directory": suite_dir,
    }[reference_kind]
    monkeypatch.setattr(
        "lm_eval.tasks.TaskManager",
        lambda **kwargs: TaskManager(include_defaults=False, **kwargs),
    )
    config = EvaluatorConfig(tasks=[str(reference)])
    manager = config.process_tasks()

    loaded = manager.load(config.tasks)

    assert loaded["group_map"] == {
        "inline_suite": ["inline_suite::config_leaf_a", "inline_suite::config_leaf_b"]
    }
    assert all(len(task.test_docs()) == 2 for task in loaded["tasks"].values())


@pytest.mark.parametrize("python_task", [False, True])
def test_task_dict_controls(local_tasks, python_task):
    _, configs = local_tasks
    config = copy.deepcopy(configs[0])
    if python_task:
        config.update({"class": LocalPythonTask, "group": "legacy_group"})
    original = copy.deepcopy(config)
    manager = TaskManager(include_defaults=False, metadata={"runtime": True})

    loaded = manager.load(config)

    assert config == original
    assert list(loaded["tasks"]) == ["config_leaf_a"]
    assert loaded["tasks"]["config_leaf_a"].config.metadata["runtime"] is True
    assert isinstance(loaded["tasks"]["config_leaf_a"], LocalPythonTask) is python_task
