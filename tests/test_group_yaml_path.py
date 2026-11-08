from pathlib import Path

import pytest
import yaml

from lm_eval.tasks._index import Kind, TaskIndex
from lm_eval.tasks.manager import TaskManager


@pytest.mark.parametrize(
    ("filename", "name"),
    [("group.yaml", "test_group"), ("include_group.yaml", "include_group")],
)
@pytest.mark.parametrize("absolute", [False, True])
def test_group_file_path_matches_registered_name(filename, name, absolute):
    configs = Path(__file__).parent / "test_configs"
    manager = TaskManager(include_path=configs, include_defaults=False)
    path = configs / filename
    if not absolute:
        path = path.relative_to(Path.cwd())

    by_name = manager.load(name)
    by_path = manager.load(str(path))

    assert by_path["group_map"] == by_name["group_map"]
    assert list(by_path["groups"]) == [name]
    assert list(by_path["tasks"]) == list(by_name["tasks"])
    for task_name, task in by_path["tasks"].items():
        assert task.config.num_fewshot == by_name["tasks"][task_name].config.num_fewshot


@pytest.mark.parametrize(
    ("config", "kind", "name"),
    [
        ({"group": "suite", "task": ["leaf"]}, Kind.GROUP, "suite"),
        ({"group": "suite", "task": "leaf"}, Kind.GROUP, "suite"),
        ({"group": "suite", "task": []}, Kind.GROUP, "suite"),
        ({"task": "leaf"}, Kind.TASK, "leaf"),
        (
            {"task": "leaf", "class": "custom.Task", "group": "legacy"},
            Kind.PY_TASK,
            "leaf",
        ),
    ],
)
def test_file_entry_name_follows_its_kind(tmp_path, config, kind, name):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    entry = TaskIndex.entry_from_path(path)

    assert entry.kind is kind
    assert entry.name == name
    assert entry.yaml_path == path.resolve()


@pytest.mark.parametrize(
    ("filename", "name", "collection"),
    [
        ("empty_group.yaml", "empty_group", "groups"),
        ("simple_task.yaml", "simple_task", "tasks"),
    ],
)
def test_other_file_specs_keep_their_names(filename, name, collection):
    configs = Path(__file__).parent / "test_configs"
    manager = TaskManager(include_path=configs, include_defaults=False)

    result = manager.load(str(configs / filename))

    assert list(result[collection]) == [name]
