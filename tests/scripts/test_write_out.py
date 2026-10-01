"""Exercise prompt export through real task loading with local, offline data."""

import json
import sys
from functools import partial

import pytest
import yaml

from lm_eval.tasks import TaskManager
from scripts import write_out


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
        "training_split": "train",
        "validation_split": "validation",
        "test_split": "test",
        "output_type": "generate_until",
        "doc_to_text": "{{question}}:",
        "doc_to_target": "{{answer}}",
        "num_fewshot": 1,
        "metric_list": [
            {"metric": "exact_match", "aggregation": "mean", "higher_is_better": True}
        ],
    }
    (task_dir / "task.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (task_dir / "group.yaml").write_text(
        yaml.safe_dump({"group": "local_group", "task": ["local_export"]}),
        encoding="utf-8",
    )
    # Keep real task discovery/loading, without scanning unrelated built-in tasks.
    monkeypatch.setattr(
        write_out, "TaskManager", partial(TaskManager, include_defaults=False)
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    return task_dir


def run_export(monkeypatch, task_dir, output, *args):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "write_out.py",
            "--tasks",
            "local_export",
            "--include_path",
            str(task_dir),
            "--output_path",
            str(output),
            *args,
        ],
    )
    write_out.main()
    return (output / "local_export").read_text(encoding="utf-8")


@pytest.mark.parametrize("selection", ["local_export", "local_group", "all_tasks"])
def test_write_out_exports_task_group_and_all_tasks(
    local_tasks, tmp_path, monkeypatch, selection
):
    text = run_export(
        monkeypatch,
        local_tasks,
        tmp_path / "out",
        "--tasks",
        selection,
        "--num_fewshot",
        "0",
    )
    assert text == "!!@@##@@!! -- Example 0\nvalidation Q0:\n"


def test_write_out_combines_splits_in_requested_order(
    local_tasks, tmp_path, monkeypatch
):
    text = run_export(
        monkeypatch,
        local_tasks,
        tmp_path / "out",
        "--sets",
        "test,train",
        "--num_fewshot",
        "0",
        "--num_examples",
        "10",
    )
    assert text.count("!!@@##@@!! -- Example") == 10
    assert "Example 7\ntest Q7:\n" in text
    assert "Example 8\ntrain Q0:\n" in text
    assert "Example 9\ntrain Q1:\n" in text


@pytest.mark.parametrize("num_examples", ["0", "-1"])
def test_write_out_nonpositive_limit_exports_all_rows(
    local_tasks, tmp_path, monkeypatch, num_examples
):
    text = run_export(
        monkeypatch,
        local_tasks,
        tmp_path / "out",
        "--num_examples",
        num_examples,
        "--num_fewshot",
        "0",
    )
    assert text.count("!!@@##@@!! -- Example") == 8
    assert text.endswith("validation Q7:\n")


def test_write_out_reports_unavailable_split(local_tasks, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="no splits which match"):
        run_export(monkeypatch, local_tasks, tmp_path / "out", "--sets", "missing")


def test_write_out_seed_controls_fewshot_prompts(local_tasks, tmp_path, monkeypatch):
    first = run_export(
        monkeypatch,
        local_tasks,
        tmp_path / "first",
        "--seed",
        "42",
        "--num_examples",
        "3",
    )
    again = run_export(
        monkeypatch,
        local_tasks,
        tmp_path / "again",
        "--seed",
        "42",
        "--num_examples",
        "3",
    )
    other = run_export(
        monkeypatch,
        local_tasks,
        tmp_path / "other",
        "--seed",
        "7",
        "--num_examples",
        "3",
    )
    assert first == again
    assert first != other
    assert "train Q" in first
    assert "validation Q0:" in first
