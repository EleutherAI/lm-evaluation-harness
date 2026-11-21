from unittest.mock import Mock

import pytest
from sqlitedict import SqliteDict

from lm_eval.api import model
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM, CachingLM, hash_args


@pytest.fixture
def cached_lm(tmp_path):
    wrapper = CachingLM(Mock(spec=LM), str(tmp_path / "responses.db"))
    try:
        yield wrapper
    finally:
        wrapper.dbdict.close()


@pytest.mark.parametrize(
    ("method", "args", "value"),
    [
        ("generate_until", ("prompt", {}), ""),
        ("loglikelihood", ("prompt", " continuation"), (-1.0, False)),
        ("loglikelihood_rolling", ("prompt",), 0.0),
    ],
)
def test_reads_existing_sqlitedict_cache(tmp_path, method, args, value):
    path = str(tmp_path / "existing.db")
    with SqliteDict(path) as db:
        db[hash_args(method, args)] = value
        db.commit()

    request = Instance(method, {}, args, 0)
    wrapper = CachingLM(Mock(spec=LM), path)
    try:
        assert getattr(wrapper, method)([request, request]) == [value, value]
        getattr(wrapper.lm, method).assert_not_called()
    finally:
        wrapper.dbdict.close()


def test_batches_reads_and_preserves_hit_and_miss_order(cached_lm, monkeypatch):
    monkeypatch.setattr(model, "_CACHE_LOOKUP_BATCH_SIZE", 3)
    a, b, c, d = [Instance("generate_until", {}, (prompt, {}), 0) for prompt in "abcd"]
    for req, response in [(a, "A"), (c, "C")]:
        cached_lm.get_cache_hook().add_partial("generate_until", req.args, response)
    cached_lm.lm.generate_until.return_value = ["B1", "D", "B2"]
    conn = cached_lm.dbdict.conn
    select = Mock(wraps=conn.select)
    monkeypatch.setattr(conn, "select", select)

    requests = [c, b, c, a, d, b, a]
    assert cached_lm.generate_until(requests) == ["C", "B1", "C", "A", "D", "B2", "A"]
    cached_lm.lm.generate_until.assert_called_once_with([b, d, b])
    reads = [args for args, _ in select.call_args_list if args[0].startswith("SELECT")]
    assert [len(params) for _, params in reads] == [2, 3, 1]

    cached_lm.lm.generate_until.reset_mock()
    assert cached_lm.generate_until(requests) == ["C", "B2", "C", "A", "D", "B2", "A"]
    cached_lm.lm.generate_until.assert_not_called()


def test_repeated_sampled_requests_bypass_existing_cache(
    cached_lm, monkeypatch, caplog
):
    monkeypatch.setattr(model, "_CACHE_LOOKUP_BATCH_SIZE", 2)
    sampled = Instance("generate_until", {}, ("prompt", {"do_sample": True}), 0)
    greedy = Instance("generate_until", {}, ("prompt", {}), 0)
    for req, response in [(sampled, "old sample"), (greedy, "greedy")]:
        cached_lm.get_cache_hook().add_partial("generate_until", req.args, response)
    cached_lm.lm.generate_until.return_value = ["first", "second", "third"]

    assert cached_lm.generate_until([sampled, sampled, greedy, sampled]) == [
        "first",
        "second",
        "greedy",
        "third",
    ]
    cached_lm.lm.generate_until.assert_called_once_with([sampled, sampled, sampled])
    assert caplog.text.count("non-deterministic sampling") == 1


def test_empty_request_list(cached_lm):
    assert cached_lm.generate_until([]) == []
    cached_lm.lm.generate_until.assert_not_called()
