from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datasets import Dataset

import lm_eval
from lm_eval.tasks import TaskManager
from lm_eval.tasks._index import Kind, TaskIndex
from lm_eval.tasks.entr import utils

REPO_ROOT = Path(__file__).resolve().parents[1]
ENTR_TASKS = REPO_ROOT / "lm_eval/tasks/entr"


def _stable_id(path: str, corpus: str, original_id: int, row_group: int, row: int) -> str:
    payload = json.dumps(
        [
            utils.VNGRS_REVISION,
            path,
            str(corpus),
            str(original_id),
            row_group,
            row,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _historical_ids_sha256(corpus: str, ids: list[str]) -> str:
    digest = hashlib.sha256()
    for stable_id in sorted(ids):
        row = (
            {"schema_version": 1, "stable_document_id": stable_id}
            if corpus == "oscar"
            else {"stable_document_id": stable_id}
        )
        options = {"ensure_ascii": False, "sort_keys": True}
        if corpus == "oscar":
            options["separators"] = (",", ":")
        digest.update((json.dumps(row, **options) + "\n").encode("utf-8"))
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(
    tmp_path: Path,
    *,
    corpus="oscar",
    bad_shard_sha=False,
    wrong_id=False,
    missing_row=False,
):
    used_path = "data/train-used.parquet"
    unused_path = "data/train-unused.parquet"
    used_file = tmp_path / "used.parquet"
    unused_file = tmp_path / "unused.parquet"
    source_rows = [
        {"text": "Türkçe bir belge.", "corpus": corpus, "original_id": 101},
        {"text": "İkinci örnek.", "corpus": corpus, "original_id": 102},
    ]
    pq.write_table(pa.Table.from_pylist(source_rows), used_file, row_group_size=1)
    pq.write_table(
        pa.Table.from_pylist(
            [{"text": "Not selected.", "corpus": "oscar", "original_id": 999}]
        ),
        unused_file,
    )

    locators = []
    for row_group, source in enumerate(source_rows):
        row = 5 if missing_row and row_group == 0 else 0
        stable_id = _stable_id(
            used_path,
            corpus,
            source["original_id"],
            row_group,
            row,
        )
        if wrong_id and row_group == 0:
            stable_id = "f" * 64
        locators.append(
            {
                "stable_document_id": stable_id,
                "corpus": corpus,
                "original_id": source["original_id"],
                "shard_path": used_path,
                "row_group": row_group,
                "row": row,
                "source_revision": utils.VNGRS_REVISION,
            }
        )
    locator_path = tmp_path / f"{corpus}_heldout.jsonl"
    _write_jsonl(locator_path, list(reversed(locators)))

    used_sha = "0" * 64 if bad_shard_sha else _sha256(used_file)
    registry = {
        "dataset": utils.VNGRS_DATASET_ID,
        "revision": utils.VNGRS_REVISION,
        "shards": [
            {
                "path": used_path,
                "revision": utils.VNGRS_REVISION,
                "bytes": used_file.stat().st_size,
                "byte_sha256": used_sha,
            },
            {
                "path": unused_path,
                "revision": utils.VNGRS_REVISION,
                "bytes": unused_file.stat().st_size,
                "byte_sha256": _sha256(unused_file),
            },
        ],
    }
    registry_path = tmp_path / "vngrs_source_shards.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    files = {
        (utils.HELDOUT_DATASET_ID, utils.LOCATOR_FILES[corpus]): locator_path,
        (utils.HELDOUT_DATASET_ID, utils.SOURCE_SHARDS_FILE): registry_path,
        (utils.VNGRS_DATASET_ID, used_path): used_file,
        (utils.VNGRS_DATASET_ID, unused_path): unused_file,
    }
    calls = []

    def fake_hf_hub_download(*, repo_id, filename, repo_type, revision):
        calls.append((repo_id, filename, repo_type, revision))
        return str(files[(repo_id, filename)])

    return {
        "locators": locators,
        "locator_path": locator_path,
        "registry_path": registry_path,
        "files": files,
        "calls": calls,
        "fake_hf_hub_download": fake_hf_hub_download,
        "used_path": used_path,
        "unused_path": unused_path,
        "source_rows": source_rows,
        "corpus": corpus,
        "bad_shard_sha": bad_shard_sha,
    }


def _patch_fixture_constants(monkeypatch, fixture):
    corpus = fixture["corpus"]
    ids = [row["stable_document_id"] for row in fixture["locators"]]
    monkeypatch.setattr(
        utils,
        "LOCATOR_SHA256",
        {**utils.LOCATOR_SHA256, corpus: _sha256(fixture["locator_path"])},
    )
    monkeypatch.setattr(
        utils,
        "SOURCE_SHARDS_SHA256",
        _sha256(fixture["registry_path"]),
    )
    monkeypatch.setattr(
        utils,
        "HISTORICAL_ID_SHA256",
        {
            **utils.HISTORICAL_ID_SHA256,
            corpus: _historical_ids_sha256(corpus, ids),
        },
    )
    monkeypatch.setattr(utils, "HELDOUT_COUNTS", {**utils.HELDOUT_COUNTS, corpus: 2})
    monkeypatch.setattr(utils, "hf_hub_download", fixture["fake_hf_hub_download"])


def test_vngrs_loader_verifies_rows_orders_ids_and_downloads_only_used_shards(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path)
    _patch_fixture_constants(monkeypatch, fixture)

    result = utils.load_vngrs_heldout(entr_corpus="oscar", version=1.0)

    rows = result["test"].to_list()
    expected_ids = sorted(row["stable_document_id"] for row in fixture["locators"])
    assert [row["stable_document_id"] for row in rows] == expected_ids
    assert {row["text"] for row in rows} == {
        row["text"] for row in fixture["source_rows"]
    }
    shard_calls = [
        filename
        for repo_id, filename, _, _ in fixture["calls"]
        if repo_id == utils.VNGRS_DATASET_ID
    ]
    assert shard_calls == [fixture["used_path"]]
    assert all(repo_type == "dataset" for _, _, repo_type, _ in fixture["calls"])
    for repo_id, _, _, revision in fixture["calls"]:
        expected_revision = (
            utils.HELDOUT_DATASET_REVISION
            if repo_id == utils.HELDOUT_DATASET_ID
            else utils.VNGRS_REVISION
        )
        assert revision == expected_revision


def test_vngrs_loader_fails_on_shard_sha256_mismatch(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, bad_shard_sha=True)
    _patch_fixture_constants(monkeypatch, fixture)

    with pytest.raises(ValueError, match="shard SHA-256 mismatch"):
        utils.load_vngrs_heldout("oscar")


def test_vngrs_loader_fails_on_recomputed_stable_id_mismatch(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, wrong_id=True)
    _patch_fixture_constants(monkeypatch, fixture)

    with pytest.raises(ValueError, match="stable document ID mismatch"):
        utils.load_vngrs_heldout("oscar")


def test_vngrs_loader_fails_on_locator_row_missing_from_parquet(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, missing_row=True)
    _patch_fixture_constants(monkeypatch, fixture)

    with pytest.raises(ValueError, match="requested rows missing"):
        utils.load_vngrs_heldout("oscar")


def test_vngrs_loader_is_corpus_generic_for_m4_locator(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, corpus="mc4")
    _patch_fixture_constants(monkeypatch, fixture)
    result = utils.load_vngrs_heldout(entr_corpus="mc4")
    assert {row["text"] for row in result["test"].to_list()} == {
        row["text"] for row in fixture["source_rows"]
    }


def test_rolling_metrics_match_wikitext_raw_text_word_and_byte_pairs():
    text = "\tTürkçe  ham metin\n"
    result = utils.process_results({"text": text}, (-7.25,))
    expected_words = len(re.split(r"\s+", text))
    expected_bytes = len(text.encode("utf-8"))
    assert result == {
        "word_perplexity": (-7.25, expected_words),
        "byte_perplexity": (-7.25, expected_bytes),
        "bits_per_byte": (-7.25, expected_bytes),
    }


def test_trwiki_process_docs_filters_empty_after_strip():
    dataset = Dataset.from_list([{"text": "  Türkçe  "}, {"text": " \n\t "}, {"text": ""}])
    processed = utils.process_docs(dataset)
    assert processed["text"] == ["  Türkçe  "]


def test_entr_task_and_group_names_are_indexed_without_loading_datasets():
    assert Path(lm_eval.__file__).resolve().is_relative_to(REPO_ROOT)
    manager = TaskManager()
    expected_tasks = {
        "entr_trwiki_tr",
        "entr_oscar_tr_heldout",
        "entr_mc4_tr_heldout",
        "entr_factual_five_choice_fixed",
        "entr_factual_five_choice_observed_error",
    }
    expected_groups = {"entr_corpus_tr", "entr_factual"}
    for name in expected_tasks | expected_groups:
        entry = manager.task_index[name]
        assert entry.yaml_path is not None
        assert entry.yaml_path.resolve().is_relative_to(ENTR_TASKS.resolve())
        if name in expected_tasks:
            assert entry.kind == Kind.TASK
        else:
            assert entry.kind == Kind.GROUP
    upstream_paths = [
        path
        for path in (REPO_ROOT / "lm_eval/tasks").iterdir()
        if path.is_dir() and path != ENTR_TASKS
    ]
    upstream_index = TaskIndex.build(upstream_paths)
    assert not (expected_tasks | expected_groups) & set(upstream_index)
