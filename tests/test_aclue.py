"""Offline regression tests for ACLUE's script-free dataset configuration."""

import csv
import io
import runpy
import sys
import zipfile
from pathlib import Path

import datasets
import pytest
import yaml

from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks._yaml_loader import load_yaml
from lm_eval.tasks.aclue._generate_configs import SUBJECTS


TASK_DIR = Path(__file__).resolve().parents[1] / "lm_eval" / "tasks" / "aclue"
COLUMNS = ("Question", "A", "B", "C", "D", "Answer")


def make_rows(subject, split, count):
    return [
        {
            "Question": f'  {subject} {split} 第{i}题，"引文"\n下一行  ',
            **{choice: f"选项{choice}，第{i}项" for choice in "ABCD"},
            "Answer": "ABCD"[i % 4],
        }
        for i in range(count)
    ]


def render_question(doc):
    return (
        f"{doc['Question'].strip()}\nA. {doc['A']}\nB. {doc['B']}"
        f"\nC. {doc['C']}\nD. {doc['D']}\n答案："
    )


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    path = tmp_path_factory.mktemp("aclue") / "aclue.zip"
    with zipfile.ZipFile(path, "w") as zip_file:
        for subject in SUBJECTS.values():
            for split, count in (("test", 2), ("dev", 5)):
                buf = io.StringIO(newline="")
                writer = csv.DictWriter(buf, fieldnames=COLUMNS)
                writer.writeheader()
                writer.writerows(make_rows(subject, split, count))
                zip_file.writestr(f"{split}/{subject}.csv", buf.getvalue())
            # The real archive contains macOS metadata that must not be loaded.
            zip_file.writestr(f"__MACOSX/dev/._{subject}.csv", b"\x00\x05\xff")
        zip_file.writestr("test/.DS_Store", b"\x00\xff")
    return path


@pytest.mark.parametrize("subject", SUBJECTS.values())
def test_aclue_loads_subject_splits_and_preserves_inputs(subject, archive, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    cfg = load_yaml(TASK_DIR / f"aclue_{subject}.yaml")
    assert cfg["dataset_name"] == subject
    assert cfg["dataset_kwargs"]["subject"] == subject
    download_and_extract = datasets.DownloadManager.download_and_extract

    def download_local_archive(manager, url):
        if not isinstance(url, str):
            return download_and_extract(manager, url)
        assert url.startswith("https://huggingface.co/datasets/tyouisen/aclue/resolve/")
        assert "/main/" not in url
        return download_and_extract(manager, str(archive))

    monkeypatch.setattr(
        datasets.DownloadManager, "download_and_extract", download_local_archive
    )
    cfg["dataset_kwargs"]["cache_dir"] = str(archive.parent / "cache")
    task = ConfigurableTask(config=cfg)
    task.set_fewshot_seed(1234)

    test_rows = make_rows(subject, "test", 2)
    dev_rows = make_rows(subject, "dev", 5)
    assert list(task.test_docs()) == test_rows
    assert list(task.fewshot_docs()) == dev_rows
    assert set(task.dataset) == {"test", "dev"}
    for split in ("test", "dev"):
        assert task.dataset[split].features == datasets.Features(
            {column: datasets.Value("string") for column in COLUMNS}
        )
    assert task.sampler.sample(5) == dev_rows
    for doc in test_rows:
        prompt = render_question(doc)
        assert task.doc_to_text(doc) == prompt
        assert task.doc_to_choice(doc) == list("ABCD")
        assert task.doc_to_target(doc) == "ABCD".index(doc["Answer"])
        for shots in (0, 5):
            expected = (
                cfg["description"]
                + "".join(
                    render_question(example) + " " + example["Answer"] + "\n\n"
                    for example in dev_rows[:shots]
                )
                + prompt
            )
            context = task.fewshot_context(doc=doc, num_fewshot=shots)
            assert context == expected
            requests = task.construct_requests(doc=doc, ctx=context)
            assert [request.request_type for request in requests] == [
                "loglikelihood"
            ] * 4
            assert [request.arguments for request in requests] == [
                (expected, f" {choice}") for choice in "ABCD"
            ]
    # Exercise every answer label using deterministic synthetic scores.
    for doc in dev_rows:
        scores = [
            (0.0 if choice == doc["Answer"] else -1.0, False) for choice in "ABCD"
        ]
        assert task.process_results(doc, scores) == {"acc": 1.0, "acc_norm": 1.0}


def test_aclue_generator_reproduces_checked_in_configs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "_generate_configs.py",
            "--base_yaml_path",
            str(TASK_DIR / "_default_template_yaml"),
        ],
    )
    runpy.run_path(str(TASK_DIR / "_generate_configs.py"), run_name="__main__")
    for subject in SUBJECTS.values():
        filename = f"aclue_{subject}.yaml"
        assert yaml.safe_load(
            (tmp_path / filename).read_text(encoding="utf-8")
        ) == yaml.safe_load((TASK_DIR / filename).read_text(encoding="utf-8"))
