import os

import pytest

from lm_eval.caching import cache


def test_save_to_cache_creates_missing_parent_directories(tmp_path, monkeypatch):
    nested_cache = tmp_path / "missing-parent" / "cache"
    monkeypatch.setattr(cache, "PATH", str(nested_cache))

    cache.save_to_cache("nested-key", {"ok": True})

    assert nested_cache.is_dir()
    assert cache.load_from_cache("nested-key", cache=True) == {"ok": True}


def test_long_cache_keys_are_hashed_below_filesystem_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "PATH", str(tmp_path))

    long_model_name = "tokenizer" + "very-long-model-name/" * 30
    cache_key = f"requests-mmlu-5shot-rank0-world_size1-chat_template-{long_model_name}"

    cache.save_to_cache(cache_key, {"ok": True})

    cache_files = os.listdir(tmp_path)
    assert len(cache_files) == 1
    assert len(cache_files[0].encode("utf-8")) <= cache.MAX_CACHE_FILENAME_BYTES
    assert cache.load_from_cache(cache_key, cache=True) == {"ok": True}


def test_short_cache_keys_keep_readable_name(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "PATH", str(tmp_path))

    cache_key = "requests-sciq-0shot-rank0-world_size1-tokenizergpt2"
    cache.save_to_cache(cache_key, ["cached"])

    cache_files = os.listdir(tmp_path)
    assert cache_files == [f"{cache_key}{cache.FILE_SUFFIX}"]
    assert cache.load_from_cache(cache_key, cache=True) == ["cached"]


def test_delete_cache_is_idempotent_when_directory_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "PATH", str(tmp_path / "never-created"))

    cache.delete_cache()

    assert not (tmp_path / "never-created").exists()


def test_delete_cache_raises_when_path_is_a_regular_file(tmp_path, monkeypatch):
    file_path = tmp_path / "not-a-dir"
    file_path.write_text("data")
    monkeypatch.setattr(cache, "PATH", str(file_path))

    try:
        cache.delete_cache()
    except NotADirectoryError:
        pass
    else:
        pytest.fail("expected NotADirectoryError when PATH is a regular file")


def test_delete_cache_removes_only_own_suffix_files(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "PATH", str(tmp_path))

    cache.save_to_cache("requests-task-key", {"ok": True})
    (tmp_path / "unrelated.txt").write_text("keep me")

    cache.delete_cache()

    assert (tmp_path / "unrelated.txt").exists()
    assert os.listdir(tmp_path) == ["unrelated.txt"]
