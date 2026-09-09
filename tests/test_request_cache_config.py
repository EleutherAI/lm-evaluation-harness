from functools import partial

import datasets
import pytest

from lm_eval.api.task import ConfigurableTask
from lm_eval.caching import cache


def render_prompt(doc, prefix):
    return prefix + doc["question"]


@pytest.fixture
def task_factory(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "PATH", str(tmp_path))

    def download(self, *args, **kwargs):
        self.dataset = datasets.DatasetDict(
            test=datasets.Dataset.from_dict(
                {"question": ["What is 2+2?"], "answer": ["4"]}
            )
        )

    monkeypatch.setattr(ConfigurableTask, "download", download)

    def make(**overrides):
        config = {
            "task": "cache_config_test",
            "dataset_path": "local",
            "test_split": "test",
            "doc_to_text": "question",
            "doc_to_target": "answer",
            "num_fewshot": 0,
            "output_type": "generate_until",
            "generation_kwargs": {"until": ["\n"], "do_sample": False},
            "metric_list": [{"metric": "exact_match"}],
        }
        config.update(overrides)
        task = ConfigurableTask(config=config)
        task.set_fewshot_seed(123)
        return task

    return make


@pytest.mark.parametrize(
    "change",
    [
        {"doc_to_text": "Updated: {{question}}"},
        {"description": "Answer concisely. "},
        {"doc_to_text": partial(render_prompt, prefix="Updated: ")},
        {"generation_kwargs": {"until": ["END"], "do_sample": False}},
        {"repeats": 2},
    ],
)
def test_changed_config_rebuilds_requests(task_factory, change):
    first = task_factory()
    first.build_all_requests(cache_requests=True)
    changed = task_factory(**change)
    changed.build_all_requests(cache_requests=True)
    uncached = task_factory(**change)
    uncached.build_all_requests(cache_requests=False)
    assert changed.instances[0].args == uncached.instances[0].args
    assert changed.instances[0].repeats == uncached.instances[0].repeats


def test_same_config_reuses_requests(task_factory, monkeypatch):
    first = task_factory()
    first.build_all_requests(cache_requests=True)
    second = task_factory(generation_kwargs={"do_sample": False, "until": ["\n"]})

    def unexpected_rebuild(*args, **kwargs):
        pytest.fail("unchanged config should hit the request cache")

    monkeypatch.setattr(second, "construct_requests", unexpected_rebuild)
    second.build_all_requests(cache_requests=True)
    assert second.instances[0].args == first.instances[0].args


def test_callable_arguments_invalidate_requests(task_factory):
    first = task_factory(doc_to_text=partial(render_prompt, prefix="First: "))
    first.build_all_requests(cache_requests=True)
    second = task_factory(doc_to_text=partial(render_prompt, prefix="Second: "))
    second.build_all_requests(cache_requests=True)
    assert second.instances[0].args[0] == "Second: What is 2+2?"


def test_legacy_cache_is_not_reused(task_factory):
    old = task_factory(doc_to_text="Old: {{question}}")
    old.build_all_requests(cache_requests=False)
    cache.save_to_cache(
        "requests-cache_config_test-0shot-rank0-world_size1-tokenizer",
        [old.instances],
    )
    current = task_factory()
    current.build_all_requests(cache_requests=True)
    assert current.instances[0].args[0] == "What is 2+2?"


def test_disabled_cache_does_not_hash_config(task_factory, monkeypatch):
    from datasets.fingerprint import Hasher

    def unexpected_hash(*args, **kwargs):
        pytest.fail("disabled request cache should not hash the config")

    task = task_factory()
    monkeypatch.setattr(Hasher, "hash", unexpected_hash)
    task.build_all_requests(cache_requests=False)
    assert task.instances[0].args[0] == "What is 2+2?"


def test_rewrite_cache_rebuilds_requests(task_factory, monkeypatch):
    first = task_factory()
    first.build_all_requests(cache_requests=True)
    second = task_factory()
    calls = []
    original = second.construct_requests

    def track_rebuild(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(second, "construct_requests", track_rebuild)
    second.build_all_requests(cache_requests=True, rewrite_requests_cache=True)
    assert calls == [True]
    assert second.instances[0].args == first.instances[0].args


@pytest.mark.parametrize("failure_stage", ["config", "hash"])
@pytest.mark.parametrize("rewrite", [False, True])
def test_fingerprint_failure_bypasses_cache(
    task_factory, monkeypatch, tmp_path, caplog, failure_stage, rewrite
):
    from datasets.fingerprint import Hasher

    old = task_factory(doc_to_text="Old: {{question}}")
    old.build_all_requests(cache_requests=False)
    cache.save_to_cache(
        "requests-cache_config_test-0shot-rank0-world_size1-tokenizer",
        [old.instances],
    )
    cached_files = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    current = task_factory(doc_to_text="New: {{question}}")

    def cannot_fingerprint(*args, **kwargs):
        raise TypeError("unsupported configuration value")

    if failure_stage == "config":
        monkeypatch.setattr(current.config, "to_dict", cannot_fingerprint)
    else:
        monkeypatch.setattr(Hasher, "hash", cannot_fingerprint)

    def unexpected_write(*args, **kwargs):
        pytest.fail("failed fingerprint must not write a cache entry")

    monkeypatch.setattr("lm_eval.api.task.save_to_cache", unexpected_write)
    current.build_all_requests(
        cache_requests=True, rewrite_requests_cache=rewrite, limit=1
    )

    assert current.instances[0].args[0] == "New: What is 2+2?"
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == cached_files
    assert "Skipping request cache" in caplog.text
    assert "cache_config_test" in caplog.text
