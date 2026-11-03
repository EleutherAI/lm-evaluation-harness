from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download

HELDOUT_DATASET_ID = "umutyesildal/entr-eval-heldout-ids"
HELDOUT_DATASET_REVISION = "427c5018d34e9e7864cbb77ab1e93cba2560ec69"
VNGRS_DATASET_ID = "vngrs-ai/vngrs-web-corpus"
VNGRS_REVISION = "ee5c6201ee84457a18182bfc483a7d8a7f3655ba"
SOURCE_SHARDS_FILE = "data/vngrs_source_shards.json"
SOURCE_SHARDS_SHA256 = "02f1717d6c0893c434d89744a1803ffcc9aeb672632dd622c91edf9c9c65a2b1"
VNGRS_COLUMNS = ("text", "corpus", "original_id")
LOCATOR_FILES = {
    "oscar": "data/oscar_heldout.jsonl",
    "mc4": "data/mc4_heldout.jsonl",
}
LOCATOR_SHA256 = {
    "oscar": "91ea0933f38723fd105210934b4a9b62cad5f861211411ad9f0b0aab6e4d5f05",
    "mc4": "c590f0ed3d7abc19df3f67cb1681b35c4bb1d7f8d56aedd3dbf40534886d1bcb",
}
HISTORICAL_ID_SHA256 = {
    "oscar": "dc30629fe76ca722745fcd0148021de6c042218ed2994281d52d13c06011dc91",
    "mc4": "485ceb3c10d841a90af2107d51ff1446ad56b4097fe68144b4aac14bb6f2d531",
}
HELDOUT_COUNTS = {"oscar": 10_000, "mc4": 10_000}


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {line_number} is not an object")
            rows.append(row)
    return rows


def _stable_document_id(
    revision: str,
    shard_path: str,
    corpus: Any,
    original_id: Any,
    row_group: int,
    row: int,
) -> str:
    payload = json.dumps(
        [revision, shard_path, str(corpus), str(original_id), row_group, row],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _historical_id_file_sha256(corpus: str, stable_ids: list[str]) -> str:
    digest = hashlib.sha256()
    for stable_id in stable_ids:
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


def load_vngrs_heldout(
    entr_corpus: str = "oscar", **_unused_metadata: Any
) -> DatasetDict:
    if entr_corpus not in LOCATOR_FILES:
        raise ValueError(f"Unsupported ENTR corpus: {entr_corpus!r}")

    locator_path = Path(
        hf_hub_download(
            repo_id=HELDOUT_DATASET_ID,
            filename=LOCATOR_FILES[entr_corpus],
            repo_type="dataset",
            revision=HELDOUT_DATASET_REVISION,
        )
    )
    manifest_path = Path(
        hf_hub_download(
            repo_id=HELDOUT_DATASET_ID,
            filename=SOURCE_SHARDS_FILE,
            repo_type="dataset",
            revision=HELDOUT_DATASET_REVISION,
        )
    )
    if _sha256_file(locator_path) != LOCATOR_SHA256[entr_corpus]:
        raise ValueError(f"{entr_corpus} locator file SHA-256 mismatch")
    if _sha256_file(manifest_path) != SOURCE_SHARDS_SHA256:
        raise ValueError("VNGRS source-shard registry SHA-256 mismatch")

    locator_rows = _read_jsonl(locator_path)
    if len(locator_rows) != HELDOUT_COUNTS[entr_corpus]:
        raise ValueError(
            f"{entr_corpus} held-out count mismatch: "
            f"expected {HELDOUT_COUNTS[entr_corpus]}, got {len(locator_rows)}"
        )
    locator_rows.sort(key=lambda row: row["stable_document_id"])
    stable_ids = [row["stable_document_id"] for row in locator_rows]
    if len(set(stable_ids)) != len(stable_ids):
        raise ValueError(f"{entr_corpus} locator contains duplicate stable IDs")
    if _historical_id_file_sha256(entr_corpus, stable_ids) != HISTORICAL_ID_SHA256[
        entr_corpus
    ]:
        raise ValueError(f"{entr_corpus} historical held-out ID hash mismatch")

    shards_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        shards_manifest.get("dataset") != VNGRS_DATASET_ID
        or shards_manifest.get("revision") != VNGRS_REVISION
        or not isinstance(shards_manifest.get("shards"), list)
    ):
        raise ValueError("VNGRS source-shard registry identity mismatch")
    shards = {}
    for entry in shards_manifest["shards"]:
        shard_path = entry.get("path")
        if not isinstance(shard_path, str) or shard_path in shards:
            raise ValueError("VNGRS shard registry has an invalid or duplicate path")
        if entry.get("revision") != VNGRS_REVISION:
            raise ValueError(f"{shard_path}: VNGRS revision mismatch")
        if (
            not isinstance(entry.get("bytes"), int)
            or isinstance(entry.get("bytes"), bool)
            or entry["bytes"] <= 0
            or not isinstance(entry.get("byte_sha256"), str)
            or len(entry["byte_sha256"]) != 64
        ):
            raise ValueError(f"{shard_path}: invalid byte identity metadata")
        shards[shard_path] = entry

    by_shard: dict[str, dict[int, dict[int, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    locator_keys = set()
    for locator in locator_rows:
        shard_path = locator.get("shard_path")
        row_group = locator.get("row_group")
        row_index = locator.get("row")
        if (
            not isinstance(shard_path, str)
            or shard_path not in shards
            or not isinstance(row_group, int)
            or isinstance(row_group, bool)
            or row_group < 0
            or not isinstance(row_index, int)
            or isinstance(row_index, bool)
            or row_index < 0
        ):
            raise ValueError("Held-out locator has an invalid source row location")
        if locator.get("source_revision") != VNGRS_REVISION:
            raise ValueError(f"{locator['stable_document_id']}: locator revision mismatch")
        if locator.get("corpus") != entr_corpus or locator.get("original_id") is None:
            raise ValueError(f"{locator['stable_document_id']}: locator source identity mismatch")
        key = (shard_path, row_group, row_index)
        if key in locator_keys:
            raise ValueError(f"Duplicate held-out source row: {key}")
        locator_keys.add(key)
        by_shard[shard_path][row_group][row_index] = locator

    text_by_id: dict[str, str] = {}
    for shard_path in sorted(by_shard):
        shard_info = shards[shard_path]
        local_path = Path(
            hf_hub_download(
                repo_id=VNGRS_DATASET_ID,
                filename=shard_path,
                repo_type="dataset",
                revision=VNGRS_REVISION,
            )
        )
        if local_path.stat().st_size != shard_info["bytes"]:
            raise ValueError(f"{shard_path}: VNGRS shard byte-length mismatch")
        if _sha256_file(local_path) != shard_info["byte_sha256"]:
            raise ValueError(f"{shard_path}: VNGRS shard SHA-256 mismatch")

        parquet = pq.ParquetFile(local_path)
        if tuple(parquet.schema_arrow.names) != VNGRS_COLUMNS:
            raise ValueError(f"{shard_path}: VNGRS logical Parquet schema mismatch")
        for row_group, requested_rows in sorted(by_shard[shard_path].items()):
            if row_group >= parquet.num_row_groups:
                raise ValueError(f"{shard_path}: missing row group {row_group}")
            found_rows = set()
            row_index = 0
            for batch in parquet.iter_batches(
                batch_size=4096,
                row_groups=[row_group],
                columns=list(VNGRS_COLUMNS),
                use_threads=False,
            ):
                for record in batch.to_pylist():
                    locator = requested_rows.get(row_index)
                    if locator is not None:
                        text = record.get("text")
                        corpus = record.get("corpus")
                        original_id = record.get("original_id")
                        if not isinstance(text, str) or not text.strip():
                            raise ValueError(
                                f"{shard_path}:{row_group}:{row_index}: empty/non-string text"
                            )
                        if (
                            corpus != entr_corpus
                            or original_id is None
                            or str(original_id) != str(locator["original_id"])
                        ):
                            raise ValueError(
                                f"{shard_path}:{row_group}:{row_index}: source row identity mismatch"
                            )
                        observed_id = _stable_document_id(
                            VNGRS_REVISION,
                            shard_path,
                            corpus,
                            original_id,
                            row_group,
                            row_index,
                        )
                        if observed_id != locator["stable_document_id"]:
                            raise ValueError(
                                f"{shard_path}:{row_group}:{row_index}: stable document ID mismatch"
                            )
                        if observed_id in text_by_id:
                            raise ValueError(f"Duplicate reconstructed stable ID: {observed_id}")
                        text_by_id[observed_id] = text
                        found_rows.add(row_index)
                    row_index += 1
            missing_rows = set(requested_rows) - found_rows
            if missing_rows:
                raise ValueError(
                    f"{shard_path}: requested rows missing from row group {row_group}: "
                    f"{sorted(missing_rows)[:5]}"
                )

    output_rows = [
        {"stable_document_id": stable_id, "text": text_by_id[stable_id]}
        for stable_id in stable_ids
        if stable_id in text_by_id
    ]
    if len(output_rows) != len(locator_rows):
        missing = sorted(set(stable_ids) - set(text_by_id))
        raise ValueError(f"Missing reconstructed held-out IDs: {missing[:5]}")
    return DatasetDict({"test": Dataset.from_list(output_rows)})


def process_docs(dataset: Dataset) -> Dataset:
    return dataset.filter(
        lambda doc: isinstance(doc.get("text"), str) and bool(doc["text"].strip())
    )


def process_results(doc: dict[str, Any], results: tuple[float, ...]) -> dict[str, tuple[float, int]]:
    (loglikelihood,) = results
    text = doc["text"]
    word_count = len(re.split(r"\s+", text))
    utf8_bytes = len(text.encode("utf-8"))
    return {
        "word_perplexity": (loglikelihood, word_count),
        "byte_perplexity": (loglikelihood, utf8_bytes),
        "bits_per_byte": (loglikelihood, utf8_bytes),
    }


def doc_to_choice(doc: dict[str, Any]) -> list[str]:
    return [candidate["completion"] for candidate in doc["candidates"]]
