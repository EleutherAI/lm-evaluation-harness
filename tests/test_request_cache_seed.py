"""Request-cache seed coverage using local data and the shipped task path."""

import random
from unittest.mock import Mock

import datasets
import pytest

from lm_eval.api.task import ConfigurableTask
from lm_eval.caching import cache


_UNSET = object()


def local_dataset():
    return datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_list(
                [{"question": f"Q{i}:", "answer": f"A{i}"} for i in range(8)]
            ),
            "test": datasets.Dataset.from_list(
                [{"question": f"Eval{i}:", "answer": f"Gold{i}"} for i in range(3)]
            ),
        }
    )


@pytest.fixture
def request_cache(tmp_path, monkeypatch):
    path = tmp_path / "requests"
    monkeypatch.setattr(cache, "PATH", str(path))
    return path


def make_task(seed=_UNSET, *, num_fewshot=3, sampler="default"):
    task = ConfigurableTask(
        config={
            "task": "local_request_cache_seed",
            "custom_dataset": local_dataset,
            "test_split": "test",
            "fewshot_split": "train",
            "fewshot_config": {"sampler": sampler},
            "doc_to_text": "question",
            "doc_to_target": "answer",
            "output_type": "loglikelihood",
            "metric_list": [],
            "num_fewshot": num_fewshot,
            "target_delimiter": " ",
            "fewshot_delimiter": "\n\n",
        }
    )
    if seed is not _UNSET:
        task.set_fewshot_seed(seed)
    return task


def build(task, *, cache_requests=False, rewrite_requests_cache=False):
    task.build_all_requests(
        cache_requests=cache_requests,
        rewrite_requests_cache=rewrite_requests_cache,
    )
    return [instance.arguments for instance in task.instances]


def expected_requests(seed, *, num_fewshot=3, sampler="default"):
    rnd = random.Random(seed)
    requests = []
    for i in range(3):
        indices = (
            list(range(num_fewshot))
            if sampler == "first_n"
            else rnd.sample(range(8), num_fewshot)
        )
        context = "".join(f"Q{j}: A{j}\n\n" for j in indices) + f"Eval{i}:"
        requests.append((context, f"Gold{i}"))
    return requests


@pytest.mark.parametrize("source_seed,target_seed", [(11, 29), (29, 11)])
def test_request_cache_respects_fewshot_seed(request_cache, source_seed, target_seed):
    cold_source = build(make_task(source_seed))
    cold_target = build(make_task(target_seed))
    assert cold_source == expected_requests(source_seed)
    assert cold_target == expected_requests(target_seed)
    assert cold_source != cold_target
    assert not request_cache.exists()

    assert build(make_task(source_seed), cache_requests=True) == cold_source
    assert list(request_cache.iterdir())
    assert build(make_task(target_seed), cache_requests=True) == cold_target


@pytest.mark.parametrize(
    "num_fewshot,sampler", [(3, "default"), (0, "default"), (3, "first_n")]
)
def test_same_seed_reuses_requests(request_cache, monkeypatch, num_fewshot, sampler):
    options = {"num_fewshot": num_fewshot, "sampler": sampler}
    expected = expected_requests(0, **options)
    assert build(make_task(0, **options), cache_requests=True) == expected
    assert list(request_cache.iterdir())

    task = make_task(0, **options)
    construct_requests = Mock(side_effect=AssertionError("cache hit rebuilt requests"))
    monkeypatch.setattr(task, "construct_requests", construct_requests)
    assert build(task, cache_requests=True) == expected
    construct_requests.assert_not_called()

    if num_fewshot == 0 or sampler == "first_n":
        assert build(make_task(29, **options)) == expected
        assert build(make_task(29, **options), cache_requests=True) == expected


@pytest.mark.parametrize("target_seed", [11, 29])
def test_refresh_rebuilds_requests(request_cache, monkeypatch, target_seed):
    build(make_task(11), cache_requests=True)
    expected = build(make_task(target_seed))
    task = make_task(target_seed)
    construct_requests = Mock(wraps=task.construct_requests)
    monkeypatch.setattr(task, "construct_requests", construct_requests)

    assert build(task, cache_requests=True, rewrite_requests_cache=True) == expected
    assert construct_requests.call_count == len(task.eval_docs)
    assert build(make_task(target_seed), cache_requests=True) == expected


def test_disabled_cache_bypasses_existing_requests(request_cache, monkeypatch):
    expected = build(make_task(29))
    assert not request_cache.exists()
    assert build(make_task(11), cache_requests=True) != expected
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in request_cache.iterdir()
    }

    task = make_task(29)
    construct_requests = Mock(wraps=task.construct_requests)
    monkeypatch.setattr(task, "construct_requests", construct_requests)
    assert build(task, cache_requests=False) == expected
    assert construct_requests.call_count == len(task.eval_docs)
    assert {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in request_cache.iterdir()
    } == before


@pytest.mark.parametrize("source_seed,target_seed", [(None, 11), (11, None)])
def test_none_and_integer_seeds_use_separate_caches(
    request_cache, monkeypatch, source_seed, target_seed
):
    build(make_task(source_seed), cache_requests=True)
    target = make_task(target_seed)
    construct_requests = Mock(wraps=target.construct_requests)
    monkeypatch.setattr(target, "construct_requests", construct_requests)
    expected = build(target, cache_requests=True)
    assert construct_requests.call_count == len(target.eval_docs)
    if target_seed is not None:
        assert expected == expected_requests(target_seed)

    # None is entropy seeded: reuse its cached realization, without requiring
    # independently constructed None tasks to draw the same or different examples.
    reused = make_task(target_seed)
    construct_requests = Mock(side_effect=AssertionError("cache hit rebuilt requests"))
    monkeypatch.setattr(reused, "construct_requests", construct_requests)
    assert build(reused, cache_requests=True) == expected
    construct_requests.assert_not_called()


def test_unset_seed_reuses_none_cache(request_cache, monkeypatch):
    expected = build(make_task(), cache_requests=True)
    for task in (make_task(), make_task(None)):
        construct_requests = Mock(
            side_effect=AssertionError("cache hit rebuilt requests")
        )
        monkeypatch.setattr(task, "construct_requests", construct_requests)
        assert build(task, cache_requests=True) == expected
        construct_requests.assert_not_called()


@pytest.mark.parametrize("seed", [1234, None])
def test_legacy_fewshot_cache_is_not_reused(request_cache, monkeypatch, seed):
    legacy_task = make_task(11)
    build(legacy_task)
    # This is the pre-seed cache key. Its entries do not identify which seed
    # generated the prompts, so even the default seed cannot safely claim them.
    legacy_key = "requests-local_request_cache_seed-3shot-rank0-world_size1-tokenizer"
    cache.save_to_cache(legacy_key, [[instance] for instance in legacy_task.instances])
    assert list(request_cache.iterdir())

    task = make_task(seed)
    construct_requests = Mock(wraps=task.construct_requests)
    monkeypatch.setattr(task, "construct_requests", construct_requests)
    actual = build(task, cache_requests=True)
    assert construct_requests.call_count == len(task.eval_docs)
    if seed is not None:
        assert actual == expected_requests(seed)


@pytest.mark.parametrize("num_fewshot", [0, None])
def test_zero_shot_cache_reused_across_seeds(request_cache, monkeypatch, num_fewshot):
    expected = expected_requests(11, num_fewshot=0)
    assert (
        build(make_task(11, num_fewshot=num_fewshot), cache_requests=True) == expected
    )
    task = make_task(29, num_fewshot=num_fewshot)
    construct_requests = Mock(side_effect=AssertionError("cache hit rebuilt requests"))
    monkeypatch.setattr(task, "construct_requests", construct_requests)
    assert build(task, cache_requests=True) == expected
    construct_requests.assert_not_called()
