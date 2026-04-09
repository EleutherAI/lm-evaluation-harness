"""Tests for the SQLite response cache (--use_cache).

Validates that cache keys include model identity so that different models
sharing the same cache DB file do not return each other's results (#2715).
"""

import pytest

from lm_eval.api.model import LM, CachingLM, hash_args


# ---------------------------------------------------------------------------
# hash_args
# ---------------------------------------------------------------------------


def test_hash_args_different_models_produce_different_hashes():
    """Two different model_ids with the same prompt must not collide."""
    args = ("What is 2+2?", "The answer is 4.")
    h1 = hash_args("loglikelihood", args, model_id="hf__pretrained=model-a")
    h2 = hash_args("loglikelihood", args, model_id="hf__pretrained=model-b")
    assert h1 != h2


def test_hash_args_same_model_produces_same_hash():
    """Same model_id and prompt must produce the same hash (cache hit)."""
    args = ("What is 2+2?", "The answer is 4.")
    h1 = hash_args("loglikelihood", args, model_id="hf__pretrained=model-a")
    h2 = hash_args("loglikelihood", args, model_id="hf__pretrained=model-a")
    assert h1 == h2


def test_hash_args_empty_model_id_differs_from_named():
    """An empty model_id and a named one must produce different hashes."""
    args = ("prompt",)
    h_empty = hash_args("loglikelihood", args, model_id="")
    h_named = hash_args("loglikelihood", args, model_id="hf__pretrained=foo")
    assert h_empty != h_named


def test_hash_args_default_model_id_is_empty():
    """Backward compat: calling hash_args without model_id defaults to ''."""
    args = ("prompt",)
    h1 = hash_args("loglikelihood", args)
    h2 = hash_args("loglikelihood", args, model_id="")
    assert h1 == h2


# ---------------------------------------------------------------------------
# CachingLM with model_id — end-to-end through the cache
# ---------------------------------------------------------------------------


class _DummyLM(LM):
    """Minimal LM stub that returns a fixed value per call."""

    def __init__(self, value):
        super().__init__()
        self._value = value

    def loglikelihood(self, requests):
        return [(self._value, True)] * len(requests)

    def loglikelihood_rolling(self, requests):
        return [self._value] * len(requests)

    def generate_until(self, requests):
        return [str(self._value)] * len(requests)


class _FakeInstance:
    """Minimal Instance stand-in carrying .args."""

    def __init__(self, args):
        self.args = args


@pytest.fixture
def cache_db(tmp_path):
    """Return a path to a temporary cache DB (cleaned up after the test)."""
    return str(tmp_path / "test_cache_rank0.db")


def test_caching_lm_different_models_do_not_collide(cache_db):
    """Two CachingLMs with different model_ids must not share cached results."""
    lm_a = _DummyLM(value=-1.0)
    lm_b = _DummyLM(value=-2.0)

    clm_a = CachingLM(lm_a, cache_db, model_id="model-a")
    clm_b = CachingLM(lm_b, cache_db, model_id="model-b")

    req = _FakeInstance(args=("context", "continuation"))

    # First call populates the cache for model-a
    res_a = clm_a.loglikelihood([req])
    assert res_a == [(-1.0, True)]

    # Second call with model-b should NOT return model-a's cached value
    res_b = clm_b.loglikelihood([req])
    assert res_b == [(-2.0, True)]


def test_caching_lm_same_model_returns_cached(cache_db):
    """Same model_id should return cached results on the second call."""
    call_count = 0

    class _CountingLM(LM):
        def loglikelihood(self, requests):
            nonlocal call_count
            call_count += len(requests)
            return [(-1.0, True)] * len(requests)

        def loglikelihood_rolling(self, requests):
            return [0.0] * len(requests)

        def generate_until(self, requests):
            return [""] * len(requests)

    lm = _CountingLM()
    clm = CachingLM(lm, cache_db, model_id="my-model")

    req = _FakeInstance(args=("context", "continuation"))

    # First call runs the model
    clm.loglikelihood([req])
    assert call_count == 1

    # Second call should hit the cache, not the model
    clm.loglikelihood([req])
    assert call_count == 1  # still 1 — no additional model call


# ---------------------------------------------------------------------------
# CacheHook with model_id
# ---------------------------------------------------------------------------


def test_cache_hook_uses_model_id(cache_db):
    """CacheHook.add_partial should include model_id in cache keys."""
    lm = _DummyLM(value=0)
    clm = CachingLM(lm, cache_db, model_id="hook-model")
    hook = clm.get_cache_hook()

    assert hook.model_id == "hook-model"

    # Write via hook
    hook.add_partial("loglikelihood", ("ctx", "cont"), (-3.0, True))

    # Verify it's stored under the model-id-aware key
    expected_hash = hash_args("loglikelihood", ("ctx", "cont"), model_id="hook-model")
    assert expected_hash in clm.dbdict
    assert clm.dbdict[expected_hash] == (-3.0, True)

    # A different model_id should NOT find this entry
    other_hash = hash_args("loglikelihood", ("ctx", "cont"), model_id="other-model")
    assert other_hash not in clm.dbdict
