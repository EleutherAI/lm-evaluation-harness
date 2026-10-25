"""Exercise scripts/make_table_tasks.py through real task loading with local data."""

import json

import pytest
import yaml

from lm_eval.tasks import TaskManager
from scripts import make_table_tasks


def _write_task(task_dir, name, splits, **overrides):
    files = {}
    for split in splits:
        path = task_dir.parent / f"{name}_{split}.jsonl"
        path.write_text(
            "".join(
                json.dumps({"question": f"{split} Q{i}", "answer": f"A{i}"}) + "\n"
                for i in range(4)
            ),
            encoding="utf-8",
        )
        files[split] = str(path)
    config = {
        "task": name,
        "dataset_path": "json",
        "dataset_kwargs": {"data_files": files},
        "output_type": "generate_until",
        "doc_to_text": "{{question}}:",
        "doc_to_target": "{{answer}}",
        "metric_list": [
            {"metric": "exact_match", "aggregation": "mean", "higher_is_better": True}
        ],
        **overrides,
    }
    if "test" in splits:
        config["test_split"] = "test"
    if "validation" in splits:
        config["validation_split"] = "validation"
    if "train" in splits:
        config["training_split"] = "train"
    (task_dir / f"{name}.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


@pytest.fixture
def task_manager(tmp_path, monkeypatch):
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    _write_task(task_dir, "with_test", ["train", "validation", "test"])
    _write_task(task_dir, "without_test", ["train", "validation"])
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    return TaskManager(include_path=str(task_dir), include_defaults=False)


def _rows(table):
    rows = []
    for line in table.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):
            continue
        rows.append(cells)
    return rows


def _row_for(table, task):
    return next(row for row in _rows(table) if row[0] == task)


def test_describes_splits_and_metrics(task_manager):
    table = make_table_tasks.make_table(task_manager, ["with_test"])

    row = _row_for(table, "with_test")
    assert row[1:4] == ["✓", "✓", "✓"]
    assert row[4] == "4"
    assert row[5] == "exact_match"


def test_counts_validation_docs_when_test_split_is_missing(task_manager):
    table = make_table_tasks.make_table(task_manager, ["without_test"])

    row = _row_for(table, "without_test")
    assert row[1:4] == ["✓", "✓", ""]
    assert row[4] == "4"


def test_writes_table_to_output(task_manager, tmp_path):
    output = tmp_path / "task_table.md"

    table = make_table_tasks.main(
        ["with_test"], output=str(output), task_manager=task_manager
    )

    assert output.read_text(encoding="utf-8") == table
    assert "with_test" in output.read_text(encoding="utf-8")


def test_check_marks_missing_split():
    assert make_table_tasks.check(True) == "✓"
    assert make_table_tasks.check(False) == " "
