"""Regression tests for cache reuse across different models.

Cache keys are derived from request arguments only (`hash_args`), so a db
populated by one model will serve its responses to another, reporting the first
model's scores for the second with no error raised. See issue #2715.
"""

import logging

import pytest

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM, MODEL_IDENTITY_KEY, CachingLM
from lm_eval.evaluator import _model_identity


class _StubLM(LM):
    """Returns a fixed response, and records how many requests reached it."""

    def __init__(self, response):
        super().__init__()
        self.response = response
        self.seen = 0

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        self.seen += len(requests)
        return [self.response] * len(requests)

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        self.seen += len(requests)
        return [0.0] * len(requests)

    def generate_until(self, requests, disable_tqdm: bool = False):
        self.seen += len(requests)
        return ["out"] * len(requests)


def _req():
    return Instance(
        request_type="loglikelihood",
        doc={},
        arguments=("ctx", "cont"),
        idx=0,
    )


def test_identity_recorded_on_first_use(tmp_path):
    db = str(tmp_path / "c.db")
    lm = CachingLM(_StubLM((1.0, True)), db, model_identity="hf(a)")
    assert lm.dbdict[MODEL_IDENTITY_KEY] == "hf(a)"


def test_reusing_cache_across_models_warns(tmp_path, caplog):
    db = str(tmp_path / "c.db")
    first = CachingLM(_StubLM((1.0, True)), db, model_identity="hf(model-a)")
    first.loglikelihood([_req()])

    with caplog.at_level(logging.WARNING):
        CachingLM(_StubLM((2.0, False)), db, model_identity="hf(model-b)")

    assert any("different model" in r.message for r in caplog.records), (
        "reusing a cache db across models must warn; without it the second model "
        "silently reports the first model's scores"
    )
    assert any(
        "model-a" in r.message and "model-b" in r.message for r in caplog.records
    )


def test_same_model_does_not_warn(tmp_path, caplog):
    db = str(tmp_path / "c.db")
    CachingLM(_StubLM((1.0, True)), db, model_identity="hf(same)")
    with caplog.at_level(logging.WARNING):
        CachingLM(_StubLM((1.0, True)), db, model_identity="hf(same)")
    assert not any("different model" in r.message for r in caplog.records)


def test_unidentifiable_model_warns_on_populated_cache(tmp_path, caplog):
    # A pre-initialized LM cannot be fingerprinted, so the cache cannot be
    # verified as belonging to it.
    db = str(tmp_path / "c.db")
    CachingLM(_StubLM((1.0, True)), db, model_identity="hf(model-a)")
    with caplog.at_level(logging.WARNING):
        CachingLM(_StubLM((2.0, False)), db, model_identity=None)
    assert any("could not be identified" in r.message for r in caplog.records)


def test_identity_key_does_not_collide_with_request_keys(tmp_path):
    # Request keys are sha256 hexdigests; the reserved key must not look like one.
    assert not all(c in "0123456789abcdef" for c in MODEL_IDENTITY_KEY)
    db = str(tmp_path / "c.db")
    lm = CachingLM(_StubLM((1.0, True)), db, model_identity="hf(a)")
    lm.loglikelihood([_req()])
    stored = [k for k in lm.dbdict.keys() if k != MODEL_IDENTITY_KEY]
    assert len(stored) == 1
    assert all(c in "0123456789abcdef" for c in stored[0])


@pytest.mark.parametrize(
    "model, model_args, expected",
    [
        ("hf", "pretrained=A", 'hf({"pretrained": "A"})'),
        ("hf", {"pretrained": "B"}, 'hf({"pretrained": "B"})'),
        ("hf", None, "hf({})"),
    ],
)
def test_model_identity_rendering(model, model_args, expected):
    assert _model_identity(model, model_args) == expected


def test_model_identity_none_for_preinitialized_model():
    assert _model_identity(_StubLM((1.0, True)), "pretrained=A") is None


def test_model_identity_distinguishes_the_issue_2715_models():
    a = _model_identity("hf", "pretrained=Qwen/Qwen2.5-3B")
    b = _model_identity("hf", "pretrained=meta-llama/Llama-3.2-3B")
    assert a != b
