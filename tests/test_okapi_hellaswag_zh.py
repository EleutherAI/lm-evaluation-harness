"""Regression tests for Chinese HellaSwag registration and bilingual endings."""

import json

import pytest
from datasets import Dataset

from lm_eval.tasks import TaskManager
from lm_eval.tasks._yaml_loader import load_yaml
from lm_eval.tasks.okapi.hellaswag_multilingual.utils import (
    process_docs,
    process_docs_zh,
)


@pytest.fixture
def normal_doc():
    return {
        "id": "synthetic-normal",
        "ctx_a": "一个人在准备早餐。",
        "ctx_b": "接着",
        "activity_label": "准备早餐",
        "endings": [" 他打开冰箱。 ", "他收起雨伞。", "他拿出面包。", "他关上书。"],
        "label": " 2",
    }


@pytest.fixture
def task_dir(pytestconfig):
    return pytestconfig.rootpath / "lm_eval/tasks/okapi/hellaswag_multilingual"


def test_normal_chinese_endings_are_preprocessed(normal_doc):
    """Keep the existing preprocessing and leading-space label behavior."""
    doc = process_docs(Dataset.from_list([normal_doc]))[0]
    assert doc["query"] == "准备早餐: 一个人在准备早餐。 接着"
    assert doc["choices"] == [
        "他打开冰箱。",
        "他收起雨伞。",
        "他拿出面包。",
        "他关上书。",
    ]
    assert doc["gold"] == 2
    assert doc["label"] == " 2"


def test_chinese_hellaswag_task_is_registered(task_dir):
    manager = TaskManager(include_path=str(task_dir), include_defaults=False)
    assert "hellaswag_vi" in manager.all_subtasks
    assert "hellaswag_zh" in manager.all_subtasks
    assert "hellaswag_zh" in manager.task_index["hellaswag_multilingual"].tags


@pytest.mark.parametrize(
    "include_bilingual", [True, False], ids=["mixed", "all-strings"]
)
def test_chinese_task_uses_builtin_loader(
    task_dir, tmp_path, normal_doc, include_bilingual
):
    """Load raw JSONL through the real YAML without losing rows or choice order."""
    last_doc = {**normal_doc, "id": "synthetic-last", "label": " 1"}
    rows = [normal_doc, last_doc]
    expected_docs = [normal_doc, last_doc]
    if include_bilingual:
        chinese_endings = [
            " 他冲洗头发。 ",
            "他涂上护发素。",
            "他打开电视。",
            "他开始跑步。",
        ]
        bilingual_doc = {**normal_doc, "id": "hellaswag/validation/8881", "label": " 3"}
        rows.insert(
            1,
            {
                **bilingual_doc,
                "endings": [
                    {"zh": ending, "en": f"English ending {index}"}
                    for index, ending in enumerate(chinese_endings)
                ],
            },
        )
        expected_docs.insert(1, {**bilingual_doc, "endings": chinese_endings})
    source = tmp_path / "val.jsonl"
    original = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    source.write_text(original, encoding="utf-8")

    config = load_yaml(task_dir / "hellaswag_zh.yaml")
    assert config["dataset_path"] == "text"
    assert not config.get("custom_dataset")
    assert config["dataset_kwargs"]["data_files"]["val"] == (
        "hf://datasets/alexandrainst/m_hellaswag@"
        "9d31dc982bd6285e081e3e3136332a38b9c1d7b7/data/zh/val.jsonl"
    )
    # Replace only the source location; use the real built-in loader and preprocessing.
    config["dataset_kwargs"] = {
        "data_files": {"val": str(source)},
        "cache_dir": str(tmp_path / "cache"),
    }
    manager = TaskManager(include_path=str(task_dir), include_defaults=False)
    task = manager.load(config)["tasks"]["hellaswag_zh"]
    assert task.config.dataset_name == "zh"
    assert task.config.validation_split == "val"
    assert not task.has_training_docs() and not task.has_test_docs()
    assert len(task.dataset["val"]) == len(rows)
    docs = list(task.validation_docs())
    assert len(docs) == len(rows)
    for expected_doc, doc in zip(expected_docs, docs, strict=True):
        assert {key: doc[key] for key in expected_doc} == expected_doc
        assert task.doc_to_text(doc) == "准备早餐: 一个人在准备早餐。 接着"
        assert task.doc_to_choice(doc) == [
            choice.strip() for choice in expected_doc["endings"]
        ]
        assert task.doc_to_target(doc) == int(expected_doc["label"])
    assert task.process_results(
        docs[-1], [(-100.0, False), (-1.0, True), (-100.0, False), (-100.0, False)]
    ) == {"acc": 1.0, "acc_norm": 1.0}
    assert source.read_text(encoding="utf-8") == original


@pytest.mark.parametrize(
    "invalid_ending",
    [{"en": "Missing Chinese"}, {"zh": 7}, 7],
    ids=["missing-zh", "non-string-zh", "invalid-type"],
)
def test_invalid_chinese_endings_report_sample(normal_doc, invalid_ending):
    """Reject unusable choices instead of dropping a row or falling back to English."""
    normal_doc["endings"][2] = invalid_ending
    raw = Dataset.from_dict({"text": [json.dumps(normal_doc, ensure_ascii=False)]})

    with pytest.raises(ValueError, match="synthetic-normal"):
        process_docs_zh(raw)
