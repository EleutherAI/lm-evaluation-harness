from __future__ import annotations

import dataclasses
import os
from typing import Any

import pytest

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM, CachingLM, hash_args


class _StubLM(LM):
    def __init__(self) -> None:
        super().__init__()
        self.counter = 0

    def loglikelihood(self, requests):
        raise NotImplementedError

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests):
        out = []
        for _ in requests:
            out.append(f"gen_{self.counter}")
            self.counter += 1
        return out


def _make_req(
    context: str,
    gen_kwargs: dict,
    *,
    task_name: str | None = "t",
    doc_id: int | None = 0,
    idx: int = 0,
    repeats: int = 1,
) -> Instance:
    return Instance(
        request_type="generate_until",
        doc={},
        arguments=(context, gen_kwargs),
        idx=idx,
        metadata=(task_name, doc_id, repeats),
    )


def _cloned(req: Instance) -> list[Instance]:
    return [dataclasses.replace(req, repeat_idx=i) for i in range(req.repeats)]


def test_deterministic_repeats_cache_k_distinct_outputs(tmp_path):
    cache_db = os.path.join(tmp_path, "cache.db")
    parent = _make_req("ctx", {"temperature": 0}, doc_id=42, repeats=4)
    clones = _cloned(parent)

    lm1 = _StubLM()
    r1 = CachingLM(lm1, cache_db).generate_until(clones)
    assert r1 == ["gen_0", "gen_1", "gen_2", "gen_3"]
    assert lm1.counter == 4

    lm2 = _StubLM()
    r2 = CachingLM(lm2, cache_db).generate_until(clones)
    assert r2 == r1
    assert lm2.counter == 0


def test_repeats_one_preserves_legacy_key_shape(tmp_path):
    cache_db = os.path.join(tmp_path, "cache.db")
    req = _make_req("ctx", {"temperature": 0}, repeats=1)

    wrapped = CachingLM(_StubLM(), cache_db)
    wrapped.generate_until([req])

    legacy_key = hash_args("generate_until", req.args)
    assert legacy_key in wrapped.dbdict
    assert wrapped.dbdict[legacy_key] == "gen_0"


@pytest.mark.parametrize(
    "gen_kwargs",
    [
        {"do_sample": True},
        {"temperature": 0.7},
        {"top_p": 0.9},
        {"top_k": 50},
    ],
)
def test_sampling_bypasses_cache_lookup_and_post_write(tmp_path, gen_kwargs):
    cache_db = os.path.join(tmp_path, "cache.db")
    parent = _make_req("ctx", gen_kwargs, repeats=3)
    clones = _cloned(parent)

    lm1 = _StubLM()
    wrapped1 = CachingLM(lm1, cache_db)
    assert wrapped1.generate_until(clones) == ["gen_0", "gen_1", "gen_2"]
    # stub doesn't call add_partial; what we're pinning is that the wrapper
    # itself never writes on the sampling path
    assert len(wrapped1.dbdict) == 0

    lm2 = _StubLM()
    assert CachingLM(lm2, cache_db).generate_until(clones) == [
        "gen_0",
        "gen_1",
        "gen_2",
    ]
    assert lm2.counter == 3


def test_partial_writes_preserve_crash_resumability_for_repeats_one(tmp_path):
    cache_db = os.path.join(tmp_path, "cache.db")

    class CrashingLM(_StubLM):
        def __init__(self, crash_after: int):
            super().__init__()
            self.crash_after = crash_after

        def generate_until(self, requests):
            out = []
            for req in requests:
                if self.counter >= self.crash_after:
                    raise RuntimeError("simulated API failure")
                out.append(f"gen_{self.counter}")
                self.cache_hook.add_partial("generate_until", req.args, out[-1])
                self.counter += 1
            return out

    reqs = [
        _make_req("ctx-a", {"temperature": 0}, doc_id=0, repeats=1),
        _make_req("ctx-b", {"temperature": 0}, doc_id=1, repeats=1),
        _make_req("ctx-c", {"temperature": 0}, doc_id=2, repeats=1),
    ]

    wrapped1 = CachingLM(CrashingLM(crash_after=2), cache_db)
    with pytest.raises(RuntimeError):
        wrapped1.generate_until(reqs)
    assert len(wrapped1.dbdict) == 2

    lm2 = _StubLM()
    r2 = CachingLM(lm2, cache_db).generate_until(reqs)
    assert r2 == ["gen_0", "gen_1", "gen_0"]
    assert lm2.counter == 1


def test_do_sample_false_alone_is_greedy_and_cached(tmp_path):
    cache_db = os.path.join(tmp_path, "cache.db")
    clones = _cloned(_make_req("ctx", {"do_sample": False}, repeats=2))

    r1 = CachingLM(_StubLM(), cache_db).generate_until(clones)

    lm2 = _StubLM()
    assert CachingLM(lm2, cache_db).generate_until(clones) == r1
    assert lm2.counter == 0


def test_do_sample_false_with_temperature_still_bypasses_cache(tmp_path):
    # API backends drop do_sample and forward temperature; must not replay.
    cache_db = os.path.join(tmp_path, "cache.db")
    clones = _cloned(
        _make_req("ctx", {"do_sample": False, "temperature": 0.7}, repeats=2)
    )

    wrapped1 = CachingLM(_StubLM(), cache_db)
    assert wrapped1.generate_until(clones) == ["gen_0", "gen_1"]
    assert len(wrapped1.dbdict) == 0

    lm2 = _StubLM()
    assert CachingLM(lm2, cache_db).generate_until(clones) == ["gen_0", "gen_1"]
    assert lm2.counter == 2


def test_explicit_temperature_zero_is_greedy_and_cached(tmp_path):
    cache_db = os.path.join(tmp_path, "cache.db")
    clones = _cloned(_make_req("ctx", {"temperature": 0}, repeats=2))

    r1 = CachingLM(_StubLM(), cache_db).generate_until(clones)

    lm2 = _StubLM()
    assert CachingLM(lm2, cache_db).generate_until(clones) == r1
    assert lm2.counter == 0


def test_repeat_idx_triggers_new_key_even_when_repeats_unset(tmp_path):
    cache_db = os.path.join(tmp_path, "cache.db")
    base = Instance(
        request_type="generate_until",
        doc={},
        arguments=("ctx", {"temperature": 0}),
        idx=0,
    )
    clones = [dataclasses.replace(base, repeat_idx=i) for i in range(3)]

    wrapped1 = CachingLM(_StubLM(), cache_db)
    r1 = wrapped1.generate_until(clones)
    assert r1 == ["gen_0", "gen_1", "gen_2"]
    assert len(wrapped1.dbdict) == 3

    lm2 = _StubLM()
    assert CachingLM(lm2, cache_db).generate_until(clones) == r1
    assert lm2.counter == 0


def test_missing_metadata_falls_back_to_occurrence_counter(tmp_path):
    cache_db = os.path.join(tmp_path, "cache.db")
    clones = _cloned(
        _make_req(
            "ctx",
            {"temperature": 0},
            task_name=None,
            doc_id=None,
            repeats=3,
        )
    )

    r1 = CachingLM(_StubLM(), cache_db).generate_until(clones)
    assert r1 == ["gen_0", "gen_1", "gen_2"]

    lm2 = _StubLM()
    assert CachingLM(lm2, cache_db).generate_until(clones) == r1
    assert lm2.counter == 0


def test_add_partial_skips_writes_on_sampled_generate_until(tmp_path):
    cache_db = os.path.join(tmp_path, "cache.db")

    class EmittingLM(_StubLM):
        def generate_until(self, requests):
            out = []
            for req in requests:
                text = f"gen_{self.counter}"
                out.append(text)
                self.cache_hook.add_partial("generate_until", req.args, text)
                self.counter += 1
            return out

    clones = _cloned(_make_req("ctx", {"temperature": 0.7}, repeats=3))
    wrapped = CachingLM(EmittingLM(), cache_db)
    wrapped.generate_until(clones)
    assert len(wrapped.dbdict) == 0


def test_loglikelihood_repeats_not_fragmented(tmp_path):
    cache_db = os.path.join(tmp_path, "cache.db")

    class LLStubLM(_StubLM):
        def loglikelihood(self, requests):
            out = []
            for _ in requests:
                out.append((float(self.counter), True))
                self.counter += 1
            return out

    base = Instance(
        request_type="loglikelihood",
        doc={},
        arguments=("ctx", "cont"),
        idx=0,
        metadata=("t", 0, 3),
    )
    clones = [dataclasses.replace(base, repeat_idx=i) for i in range(3)]

    wrapped = CachingLM(LLStubLM(), cache_db)
    wrapped.loglikelihood(clones)
    assert list(wrapped.dbdict.keys()) == [hash_args("loglikelihood", base.args)]


@pytest.mark.parametrize(
    "gen_kwargs",
    [{"top_k": 0}, {"top_k": 1}, {"top_k": 1, "temperature": 0}],
)
def test_top_k_le_one_is_not_sampling(tmp_path, gen_kwargs):
    cache_db = os.path.join(tmp_path, "cache.db")
    clones = _cloned(_make_req("ctx", gen_kwargs, repeats=2))

    r1 = CachingLM(_StubLM(), cache_db).generate_until(clones)

    lm2 = _StubLM()
    assert CachingLM(lm2, cache_db).generate_until(clones) == r1
    assert lm2.counter == 0


def test_ddp_padding_duplicates_do_not_poison_real_slots(tmp_path):
    cache_db = os.path.join(tmp_path, "cache.db")
    parent = _make_req("ctx-last", {"temperature": 0}, doc_id=7, repeats=3)
    clones = _cloned(parent)
    padding = [dataclasses.replace(c, is_padding=True) for c in _cloned(parent)]

    CachingLM(_StubLM(), cache_db).generate_until(clones + padding)

    lm2 = _StubLM()
    assert CachingLM(lm2, cache_db).generate_until(clones) == [
        "gen_0",
        "gen_1",
        "gen_2",
    ]
    assert lm2.counter == 0


def test_padding_resps_do_not_contaminate_real_request():
    # resps is shared between clones and padding via shallow copy; the
    # evaluator-side append skip keeps downstream filters seeing K entries,
    # not K + numpad*K.
    real = _make_req("ctx", {"temperature": 0}, doc_id=0, repeats=2)
    clones = _cloned(real)
    padding = [dataclasses.replace(c, is_padding=True) for c in _cloned(real)]

    # resps is the same list object on every clone and on the original
    assert all(c.resps is real.resps for c in clones + padding)

    for i, req in enumerate(clones + padding):
        if req.is_padding:
            continue
        req.resps.append(f"gen_{i}")

    assert real.resps == ["gen_0", "gen_1"]
