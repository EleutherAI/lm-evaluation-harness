"""Request-cache format compatibility and bounded serialization memory."""

import gc
import tracemalloc
from pathlib import Path

import datasets
import dill
import pytest

from lm_eval.api.instance import Instance
from lm_eval.api.task import ConfigurableTask
from lm_eval.caching import cache


@pytest.fixture
def cache_directory(tmp_path, monkeypatch):
    directory = tmp_path / "nested" / "cache"
    monkeypatch.setattr(cache, "PATH", str(directory))
    return directory


def _requests():
    doc = {"prompt": "Café 猫\nQuestion?", "answer": "yes"}
    return [
        [
            Instance(
                request_type="loglikelihood",
                doc=doc,
                arguments=(doc["prompt"], answer),
                idx=index,
                metadata=("streaming-cache", 7, 2),
            )
            for index, answer in enumerate((" yes", " no"))
        ]
    ]


@pytest.mark.parametrize("protocol", [3, 4, 5])
def test_existing_dill_cache_keeps_requests_and_aliases(cache_directory, protocol):
    requests = _requests()
    cache_directory.mkdir(parents=True)
    path = Path(cache._cache_file_path("existing"))
    path.write_bytes(dill.dumps(requests, protocol=protocol))

    restored = cache.load_from_cache("existing", cache=True)

    assert restored == requests
    assert restored[0][0].doc is restored[0][1].doc
    assert restored[0][0].arguments[0] is restored[0][0].doc["prompt"]
    assert restored[0][0].metadata == ("streaming-cache", 7, 2)


def test_new_cache_is_byte_compatible_with_existing_serializer(cache_directory):
    requests = _requests()
    expected = dill.dumps(requests)

    cache.save_to_cache("new", requests)

    assert Path(cache._cache_file_path("new")).read_bytes() == expected
    assert cache.load_from_cache("new", cache=True) == requests
    # Deserialize only the cache this test just wrote.
    assert dill.loads(Path(cache._cache_file_path("new")).read_bytes()) == requests  # noqa: S301


def test_recursive_and_callable_payloads_keep_dill_semantics(cache_directory):
    values = ["shared"]
    values.append(values)
    offset = 7
    payload = {"values": values, "alias": values, "function": lambda x: x + offset}

    cache.save_to_cache("recursive", payload)
    restored = cache.load_from_cache("recursive", cache=True)

    assert restored["values"] is restored["alias"]
    assert restored["values"][1] is restored["values"]
    assert restored["function"](5) == 12


@pytest.mark.parametrize("content", [b"", b"not a pickle", b"\x80\x04\x95\xff"])
def test_corrupted_cache_remains_a_cache_miss(cache_directory, content):
    cache_directory.mkdir(parents=True)
    Path(cache._cache_file_path("broken")).write_bytes(content)
    assert cache.load_from_cache("broken", cache=True) is None


def test_missing_or_disabled_cache_remains_a_cache_miss(cache_directory):
    assert cache.load_from_cache("absent", cache=True) is None
    assert not cache_directory.exists()
    cache.save_to_cache("disabled", _requests())
    assert cache.load_from_cache("disabled", cache=False) is None


def test_native_task_rebuilds_and_reuses_request_cache(cache_directory):
    data = datasets.DatasetDict(
        {
            "test": datasets.Dataset.from_list(
                [{"prompt": f"Question {index}", "target": "yes"} for index in range(3)]
            )
        }
    )
    config = {
        "task": "streaming-cache-task",
        "custom_dataset": lambda **kwargs: data,
        "test_split": "test",
        "doc_to_text": "prompt",
        "doc_to_target": "target",
        "output_type": "generate_until",
        "generation_kwargs": {"do_sample": False, "until": ["\n"]},
        "metric_list": [
            {"metric": "exact_match", "aggregation": "mean", "higher_is_better": True}
        ],
    }
    task = ConfigurableTask(config=config)
    task.build_all_requests(cache_requests=True)
    [path] = cache_directory.iterdir()
    serialized = path.read_bytes()
    expected = [instance.args for instance in task.instances]

    restored_task = ConfigurableTask(config=config)
    restored_task.build_all_requests(cache_requests=True, limit=2)

    assert [instance.args for instance in restored_task.instances] == expected[:2]
    assert [instance.doc_id for instance in restored_task.instances] == [0, 1]
    assert path.read_bytes() == serialized
    assert (
        restored_task.process_results(restored_task.eval_docs[0], ["yes"])[
            "exact_match"
        ]
        == 1
    )


def test_writing_large_payload_does_not_buffer_a_full_pickle(cache_directory):
    payload = b"x" * (8 * 1024 * 1024)
    gc.collect()
    tracemalloc.start()
    try:
        cache.save_to_cache("large", {"payload": payload})
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert cache.load_from_cache("large", cache=True) == {"payload": payload}
    # The serialized byte payload can go directly to the file. Allow ample
    # room for Pickler state without retaining another complete 8 MiB buffer.
    assert peak < len(payload) // 2


def test_loading_large_payload_does_not_retain_the_serialized_copy(cache_directory):
    payload = b"x" * (8 * 1024 * 1024)
    cache.save_to_cache("large", {"payload": payload})
    gc.collect()
    tracemalloc.start()
    try:
        restored = cache.load_from_cache("large", cache=True)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert restored == {"payload": payload}
    # One payload is necessary for the restored object; the complete encoded
    # file must not add another payload-sized temporary allocation.
    assert peak < len(payload) * 1.5
