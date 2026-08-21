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


import sqlite3
import tempfile

import pytest

from lm_eval.api.model import CachingLM, JsonSqliteDict


def test_json_sqlite_dict_round_trip():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name

    try:
        d = JsonSqliteDict(path)
        assert "missing" not in d

        d["a"] = (1.0, True)
        d["b"] = 2.5
        d["c"] = "hello"
        d.commit()
        d.close()

        d2 = JsonSqliteDict(path)
        assert "a" in d2
        assert d2["a"] == [1.0, True]
        assert d2["b"] == 2.5
        assert d2["c"] == "hello"
    finally:
        if path:
            import os
            os.unlink(path)


def test_json_sqlite_dict_refuses_legacy_pickle_cache():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name

    try:
        # Recreate the schema sqlitedict uses so our detector sees it.
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE unnamed (key TEXT PRIMARY KEY, value BLOB)")
        conn.commit()
        conn.close()

        d = JsonSqliteDict(path)
        assert "key" not in d
        with pytest.raises(RuntimeError):
            d["key"] = "value"
    finally:
        if path:
            import os
            os.unlink(path)


def test_caching_lm_do_sample_skips_cache(monkeypatch):
    """Non-deterministic generate_until requests should not be cached."""
    import lm_eval.api.model as model_module

    called = []

    class DummyLM:
        def __init__(self):
            self.cache_hook = model_module.CacheHook(None)

        def generate_until(self, requests):
            called.extend(requests)
            return ["generated" for _ in requests]

        def set_cache_hook(self, hook):
            self.cache_hook = hook

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = f"{tmpdir}/cache.db"
        lm = CachingLM(DummyLM(), cache_path)

        # do_sample=True should bypass cache
        from lm_eval.api.instance import Instance
        req = Instance(
            request_type="generate_until",
            doc={},
            arguments=(("prompt", {}), {"do_sample": True}),
            idx=0,
        )
        lm.generate_until([req])
        assert len(called) == 1

        # A second identical request should still hit the model because it was never cached.
        called.clear()
        lm.generate_until([req])
        assert len(called) == 1
