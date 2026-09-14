import pytest

from lm_eval.utils import _build_hierarchy_info


@pytest.mark.parametrize(
    "group_subtasks",
    [
        {"root": ["a"], "a": ["b"], "b": ["a"]},
        {"a": ["b"], "b": ["a"]},
    ],
)
def test_build_hierarchy_info_rejects_cycles(group_subtasks):
    with pytest.raises(ValueError, match="Cycle detected in group hierarchy"):
        _build_hierarchy_info(group_subtasks, set(group_subtasks))


def test_build_hierarchy_info_preserves_nested_order():
    group_subtasks = {
        "root": ["group_b", "group_a"],
        "group_a": ["task_a"],
        "group_b": ["task_b"],
    }

    depth_map, ordered = _build_hierarchy_info(
        group_subtasks, {"root", "group_a", "group_b", "task_a", "task_b"}
    )

    assert depth_map == {
        "root": 0,
        "group_a": 1,
        "task_a": 2,
        "group_b": 1,
        "task_b": 2,
    }
    assert ordered == ["root", "group_a", "task_a", "group_b", "task_b"]


def test_build_hierarchy_info_accepts_shared_children():
    group_subtasks = {
        "root": ["left", "right"],
        "left": ["shared"],
        "right": ["shared"],
    }

    depth_map, ordered = _build_hierarchy_info(group_subtasks, {"root"})

    assert depth_map == {"root": 0, "left": 1, "shared": 2, "right": 1}
    assert ordered == ["root"]
