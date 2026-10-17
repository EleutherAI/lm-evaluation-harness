"""Exercise example-prompt printing through real task loading with local, offline data."""

import json
import runpy
import sys
from functools import partial

import pytest
import yaml

from lm_eval.tasks import TaskManager
from scripts import get_prompts


@pytest.fixture
def local_tasks(tmp_path, monkeypatch):
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    files = {}
    for split in ["train", "validation", "test"]:
        path = tmp_path / f"{split}.jsonl"
        path.write_text(
            "".join(
                json.dumps({"question": f"{split} Q{i}", "answer": f"A{i}"}) + "\n"
                for i in range(8)
            ),
            encoding="utf-8",
        )
        files[split] = str(path)
    config = {
        "task": "local_export",
        "dataset_path": "json",
        "dataset_kwargs": {"data_files": files},
        "validation_split": "validation",
        "output_type": "generate_until",
        "doc_to_text": "{{question}}:",
        "doc_to_target": "{{answer}}",
        "metric_list": [
            {"metric": "exact_match", "aggregation": "mean", "higher_is_better": True}
        ],
    }
    (task_dir / "task.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    # Keep real task discovery/loading, without scanning unrelated built-in tasks.
    monkeypatch.setattr(
        get_prompts,
        "TaskManager",
        partial(TaskManager, include_defaults=False),
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    return task_dir


def test_prints_context_and_target_pairs(local_tasks, capsys):
    task_manager = TaskManager(include_path=str(local_tasks), include_defaults=False)
    get_prompts.main(["local_export"], task_manager=task_manager)
    out = capsys.readouterr().out
    assert "# local_export" in out
    assert out.count("**Context**") == 3
    assert out.count("**Target**") == 3
    assert "validation Q0" in out


def test_num_examples_limits_output(local_tasks, capsys):
    task_manager = TaskManager(include_path=str(local_tasks), include_defaults=False)
    get_prompts.main(["local_export"], num_examples=2, task_manager=task_manager)
    out = capsys.readouterr().out
    assert out.count("**Context**") == 2
    assert out.count("**Target**") == 2


def test_script_entrypoint_passes_include_path(local_tasks, capsys, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "get_prompts.py",
            "--tasks",
            "local_export",
            "--num",
            "1",
            "--include_path",
            str(local_tasks),
        ],
    )
    runpy.run_path(get_prompts.__file__, run_name="__main__")
    out = capsys.readouterr().out
    assert out.count("**Context**") == 1
