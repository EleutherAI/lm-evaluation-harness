import os

import dill
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


@pytest.mark.parametrize(
    "tokenizer_name",
    [
        "C:__models__tokenizer",  # HFLM replaces slashes in a local tokenizer path.
        r"C:\models\tokenizer",
        "http://localhost:8000/model",
    ],
)
def test_colon_keys_use_regular_files_that_can_be_cleared(
    tmp_path, monkeypatch, tokenizer_name
):
    monkeypatch.setattr(cache, "PATH", str(tmp_path))
    key = f"requests-sciq-0shot-rank0-world_size1-tokenizer{tokenizer_name}"

    cache.save_to_cache(key, {"answer": 42})

    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].name.endswith(cache.FILE_SUFFIX)
    assert ":" not in files[0].name
    assert cache.load_from_cache(key, cache=True) == {"answer": 42}

    cache.delete_cache()

    assert list(tmp_path.iterdir()) == []
    assert cache.load_from_cache(key, cache=True) is None


def test_colon_key_does_not_alias_underscore_key(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "PATH", str(tmp_path))
    cache.save_to_cache("model:revision", ["colon"])
    cache.save_to_cache("model_revision", ["underscore"])

    assert cache.load_from_cache("model:revision", cache=True) == ["colon"]
    assert cache.load_from_cache("model_revision", cache=True) == ["underscore"]


def test_long_colon_key_remains_a_bounded_regular_file(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "PATH", str(tmp_path))
    key = "requests-tokenizerC:__" + "模型" * 120

    cache.save_to_cache(key, ["cached"])

    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].name.endswith(cache.FILE_SUFFIX)
    assert ":" not in files[0].name
    assert len(files[0].name.encode("utf-8")) <= cache.MAX_CACHE_FILENAME_BYTES
    assert cache.load_from_cache(key, cache=True) == ["cached"]


def test_colon_key_does_not_reuse_legacy_path(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "PATH", str(tmp_path))
    key = "requests-tokenizerC:__models__tokenizer"
    # This creates an alternate data stream on Windows and a file elsewhere.
    legacy_path = tmp_path / f"{key}{cache.FILE_SUFFIX}"
    legacy_path.write_bytes(dill.dumps(["legacy"]))

    assert cache.load_from_cache(key, cache=True) is None


def test_clearing_local_tokenizer_cache_rebuilds_task_requests(tmp_path, monkeypatch):
    from datasets import Dataset

    from lm_eval.api.task import ConfigurableTask

    monkeypatch.setattr(cache, "PATH", str(tmp_path))
    builds = []

    class LocalTask(ConfigurableTask):
        def download(self, *args, **kwargs):
            self.dataset = {"test": Dataset.from_dict({"q": ["hello"], "a": ["world"]})}

        def construct_requests(self, *args, **kwargs):
            builds.append(self)
            return super().construct_requests(*args, **kwargs)

    config = {
        "task": "local-tokenizer-cache",
        "test_split": "test",
        "output_type": "loglikelihood",
        "doc_to_text": "q",
        "doc_to_target": "a",
        "num_fewshot": 0,
    }

    def render_chat(messages, **kwargs):
        return messages[-1]["content"]

    request_kwargs = {
        "cache_requests": True,
        "apply_chat_template": True,
        "chat_template": render_chat,
        "tokenizer_name": "C:__models__tokenizer",
    }
    first = LocalTask(config=config)
    first.build_all_requests(**request_kwargs)
    assert len(builds) == 1

    cache.delete_cache()

    second = LocalTask(config=config)
    second.build_all_requests(**request_kwargs)
    assert len(builds) == 2
