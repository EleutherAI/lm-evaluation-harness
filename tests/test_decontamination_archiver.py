"""Empty shards remain valid inputs to the decontamination readers."""

import json
from pathlib import Path

import pytest


pytest.importorskip("jsonlines")
zstandard = pytest.importorskip("zstandard")

from lm_eval.decontamination import archiver
from lm_eval.decontamination.decontaminate import get_train_overlap


@pytest.mark.parametrize("method", ["read", "read_tqdm", "read_and_tell", "read_slow"])
@pytest.mark.parametrize("records", [[], ["café 10", "東京 11"]])
def test_text_archive_reader_round_trip(tmp_path, method, records):
    path = tmp_path / "ngrams.txt"
    writer = archiver.TextArchive(str(path))
    for record in records:
        writer.add_data(record)
    writer.commit()

    reader = archiver.TextReader(str(path))
    actual = list(getattr(reader, method)())
    expected = records
    if method == "read_and_tell" and records:
        expected = [("café 10", 9), ("東京 11", 10)]
    assert actual == expected


def test_overlap_scan_continues_after_empty_shard(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    shards = tmp_path / "shards"
    shards.mkdir()
    (shards / "info.json").write_text(json.dumps({"ngram_size": 2}), encoding="utf-8")
    compressor = zstandard.ZstdCompressor()
    empty = shards / "empty.sorted.zst"
    populated = shards / "populated.sorted.zst"
    empty.write_bytes(compressor.compress(b""))
    populated.write_bytes(compressor.compress(b"alpha beta 10\n"))
    decompressed = []

    def decompress_fixture(command):
        # Replace only the external zstd executable; use the real compressed files,
        # TextReader, lookup construction, and overlap detection.
        source = Path(command.removeprefix("zstd -d "))
        assert source in (empty, populated)
        source.with_suffix("").write_bytes(
            zstandard.ZstdDecompressor().decompress(source.read_bytes())
        )
        decompressed.append(source)
        return 0

    monkeypatch.setattr(archiver.os, "system", decompress_fixture)
    monkeypatch.setattr(
        "lm_eval.decontamination.decontaminate.glob.glob",
        lambda pattern: [str(empty), str(populated)],
    )
    assert get_train_overlap(
        {("fixture", "test"): ["alpha beta", "gamma delta"]}, str(shards), 2
    ) == {"fixture": {0}}
    assert decompressed == [empty, populated]
    assert not empty.with_suffix("").exists()
    assert not populated.with_suffix("").exists()
