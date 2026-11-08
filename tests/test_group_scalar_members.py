import json

import pytest
import yaml

from lm_eval.tasks import TaskManager


@pytest.fixture
def scalar_group_tasks(tmp_path):
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
    for name in ("scalar_leaf_a", "scalar_leaf_b"):
        config = {
            "task": name,
            "tag": "scalar_leaves",
            "dataset_path": "json",
            "dataset_kwargs": {"data_files": {"test": str(data_path)}},
            "test_split": "test",
            "output_type": "multiple_choice",
            "doc_to_text": "question",
            "doc_to_choice": "choices",
            "doc_to_target": "answer",
            "num_fewshot": 0,
            "metric_list": [
                {"metric": "acc", "aggregation": "mean", "higher_is_better": True}
            ],
        }
        (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (tmp_path / "subgroup.yaml").write_text(
        yaml.safe_dump(
            {
                "group": "scalar_subgroup",
                "task": ["scalar_leaf_a", "scalar_leaf_b"],
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


@pytest.mark.parametrize(
    ("member", "expected_tasks"),
    [
        ("scalar_leaf_a", {"scalar_leaf_a"}),
        ("scalar_leaves", {"scalar_leaf_a", "scalar_leaf_b"}),
        ("scalar_subgroup", {"scalar_leaf_a", "scalar_leaf_b"}),
    ],
)
@pytest.mark.parametrize("as_list", [False, True])
def test_group_member_scalar_matches_singleton_list(
    scalar_group_tasks, member, expected_tasks, as_list
):
    config = {
        "group": "scalar_suite",
        "group_alias": "Scalar suite",
        "task": [member] if as_list else member,
        "num_fewshot": 1,
        "aggregate_metric_list": [{"metric": "acc"}],
    }
    (scalar_group_tasks / "suite.yaml").write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )
    manager = TaskManager(include_path=scalar_group_tasks, include_defaults=False)

    loaded = manager.load("scalar_suite")

    assert set(loaded["tasks"]) == expected_tasks
    group = loaded["groups"]["scalar_suite"]
    assert group.alias == "Scalar suite"
    assert group.has_aggregation
    assert group.to_dict()["task"] == [member]
    assert {task.task_name for task in group.get_all_tasks()} == expected_tasks
    assert all(task.config.num_fewshot == 1 for task in loaded["tasks"].values())


def test_inline_subgroup_scalar_keeps_its_child(scalar_group_tasks):
    config = {
        "group": "scalar_parent",
        "task": [
            {"group": "inline", "task": "scalar_leaf_a"},
            "scalar_leaf_b",
        ],
    }
    (scalar_group_tasks / "parent.yaml").write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )
    manager = TaskManager(include_path=scalar_group_tasks, include_defaults=False)

    loaded = manager.load("scalar_parent")

    assert set(loaded["tasks"]) == {"scalar_leaf_a", "scalar_leaf_b"}
    assert loaded["group_map"] == {
        "scalar_parent": ["scalar_parent::inline", "scalar_leaf_b"],
        "scalar_parent::inline": ["scalar_leaf_a"],
    }


def test_legacy_loader_expands_scalar_tag(scalar_group_tasks):
    (scalar_group_tasks / "suite.yaml").write_text(
        "group: scalar_suite\ntask: scalar_leaves\n", encoding="utf-8"
    )
    manager = TaskManager(include_path=scalar_group_tasks, include_defaults=False)

    with pytest.warns(DeprecationWarning):
        loaded = manager.load_task_or_group("scalar_suite")

    assert len(loaded) == 1
    assert set(next(iter(loaded.values()))) == {"scalar_leaf_a", "scalar_leaf_b"}


@pytest.mark.parametrize("members", [None, []])
def test_empty_group_still_has_no_children(tmp_path, members):
    (tmp_path / "empty.yaml").write_text(
        yaml.safe_dump({"group": "empty", "task": members}), encoding="utf-8"
    )
    manager = TaskManager(include_path=tmp_path, include_defaults=False)

    loaded = manager.load("empty")

    assert loaded["tasks"] == {}
    assert loaded["groups"]["empty"].child_names == []
