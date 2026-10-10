from textwrap import dedent

from lm_eval.tasks import TaskManager


def _write_task(tmp_path, name, tag=None):
    tag_line = f"tag: {tag}\n" if tag else ""
    (tmp_path / f"{name}.yaml").write_text(
        f"""task: {name}
custom_dataset: !function group_override_data.custom_dataset
output_type: multiple_choice
test_split: test
doc_to_text: "{{{{question}}}}"
doc_to_choice: ["A", "B"]
doc_to_target: gold
{tag_line}metric_list:
  - metric: acc
"""
    )


def test_task_reference_overrides_preserve_group_and_tag_leaf_names(tmp_path):
    (tmp_path / "group_override_data.py").write_text(
        dedent(
            """
            import datasets

            def custom_dataset(**kwargs):
                return {
                    "test": datasets.Dataset.from_list(
                        [{"question": "question", "gold": 0}]
                    )
                }
            """
        )
    )
    _write_task(tmp_path, "task_a")
    _write_task(tmp_path, "task_b", tag="my_tag")
    _write_task(tmp_path, "task_c", tag="my_tag")
    (tmp_path / "inner.yaml").write_text("group: inner\ntask: [task_a, task_b]\n")
    (tmp_path / "outer.yaml").write_text(
        "group: outer\ntask:\n  - task: inner\n    num_fewshot: 2\n"
    )
    (tmp_path / "outer_tag.yaml").write_text(
        "group: outer_tag\ntask:\n  - task: my_tag\n    num_fewshot: 2\n"
    )

    manager = TaskManager(include_path=tmp_path, include_defaults=False)

    group_tasks = manager.load(["outer"])["tasks"]
    assert set(group_tasks) == {"task_a", "task_b"}
    assert all(task.config.num_fewshot == 2 for task in group_tasks.values())

    tag_tasks = manager.load(["outer_tag"])["tasks"]
    assert set(tag_tasks) == {"task_b", "task_c"}
    assert all(task.config.num_fewshot == 2 for task in tag_tasks.values())
