import os

from lm_eval.caching import cache


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
