import json
import zipfile
from pathlib import Path

from lm_eval.tasks.scrolls.task import (
    _drop_duplicates_in_input,
    _load_scrolls_dataset_from_zip,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_load_scrolls_dataset_from_zip(tmp_path: Path):
    dataset_name = "qasper"
    root = tmp_path / dataset_name
    root.mkdir()

    train_rows = [
        {
            "id": "a",
            "pid": "a_0",
            "input": "q1?\n\ntext one",
            "output": "ans1",
        }
    ]
    validation_rows = [
        {
            "id": "b",
            "pid": "b_0",
            "input": "q2?\n\ntext two",
            "output": "ans2a",
        },
        {
            "id": "b",
            "pid": "b_1",
            "input": "q2?\n\ntext two",
            "output": "ans2b",
        },
        {
            "id": "c",
            "pid": "c_0",
            "input": "hard?\n\ntext three",
            "output": "ans3",
            "is_hard": True,
        },
    ]
    test_rows = [
        {
            "id": "d",
            "pid": "d_0",
            "input": "q4?\n\ntest",
            "output": "",
        }
    ]

    _write_jsonl(root / "train.jsonl", train_rows)
    _write_jsonl(root / "validation.jsonl", validation_rows)
    _write_jsonl(root / "test.jsonl", test_rows)

    zip_path = tmp_path / f"{dataset_name}.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for split in ("train", "validation", "test"):
            zf.write(root / f"{split}.jsonl", arcname=f"{dataset_name}/{split}.jsonl")

    dataset = _load_scrolls_dataset_from_zip(str(zip_path), dataset_name)

    assert set(dataset.keys()) == {"train", "validation", "test"}
    assert len(dataset["train"]) == 1
    assert len(dataset["validation"]) == 3
    assert len(dataset["test"]) == 1
    assert dataset["train"][0]["id"] == "a"
    assert "is_hard" not in dataset["validation"][2]
    assert set(dataset["validation"].column_names) == {
        "id",
        "pid",
        "input",
        "output",
    }

    deduped = _drop_duplicates_in_input(dataset["validation"])
    assert len(deduped) == 2
    assert deduped[0]["outputs"] == ["ans2a", "ans2b"]
    assert deduped[1]["outputs"] == ["ans3"]
