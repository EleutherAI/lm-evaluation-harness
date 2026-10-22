import os
from functools import partial

import pytest
from datasets import Dataset

from lm_eval.api.task import ConfigurableTask
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


@pytest.mark.parametrize(
    "first_name,second_name",
    [
        ("org/model", "org_model"),
        (r"org\model", "org_model"),
        ("org//model", "org_model"),
        ("org/model", r"org\model"),
        ("org/model", "org//model"),
    ],
)
def test_sanitized_cache_keys_do_not_alias(
    tmp_path, monkeypatch, first_name, second_name
):
    monkeypatch.setattr(cache, "PATH", str(tmp_path))
    prefix = "requests-sciq-0shot-rank0-world_size1-tokenizer"
    first_key, second_key = prefix + first_name, prefix + second_name

    cache.save_to_cache(first_key, ["first"])
    assert cache.load_from_cache(second_key, cache=True) is None

    cache.save_to_cache(second_key, ["second"])
    assert cache.load_from_cache(first_key, cache=True) == ["first"]
    assert cache.load_from_cache(second_key, cache=True) == ["second"]
    assert len(os.listdir(tmp_path)) == 2


def test_separator_key_does_not_reuse_ambiguous_legacy_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "PATH", str(tmp_path))
    # The old filename cannot establish whether the key contained a separator.
    cache.save_to_cache("org_model", ["legacy"])

    assert cache.load_from_cache("org/model", cache=True) is None
    # Unchanged short keys retain their old paths. A cache polluted by an old
    # separator collision must therefore be cleared; its origin is unknowable.
    assert cache.load_from_cache("org_model", cache=True) == ["legacy"]


@pytest.mark.parametrize("name", ["org/model", r"org\model"])
def test_request_cache_keeps_custom_tokenizer_contexts_separate(
    tmp_path, monkeypatch, name
):
    monkeypatch.setattr(cache, "PATH", str(tmp_path))

    class LocalTask(ConfigurableTask):
        def download(self, *args, **kwargs):
            self.dataset = {
                "test": Dataset.from_list([{"question": "Hello", "answer": "world"}])
            }

    def render(messages, *, tokenizer_name, **kwargs):
        return f"{tokenizer_name}: {messages[-1]['content']}"

    config = {
        "task": "cache-test",
        "test_split": "test",
        "output_type": "loglikelihood",
        "doc_to_text": "question",
        "doc_to_target": "answer",
        "num_fewshot": 0,
    }
    for tokenizer_name in [name, "org_model", name]:
        task = LocalTask(config=config)
        task.build_all_requests(
            cache_requests=True,
            apply_chat_template=True,
            tokenizer_name=tokenizer_name,
            chat_template=partial(render, tokenizer_name=tokenizer_name),
        )
        assert task.instances[0].args[0] == f"{tokenizer_name}: Hello"


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
