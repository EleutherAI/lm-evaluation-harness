"""Tests for the `--use_cache` response cache (`lm_eval.api.model.CachingLM`)."""

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM, CachingLM, hash_args


class StubLM(LM):
    """An LM that answers with fixed values and counts how often it is asked."""

    def __init__(self, loglikelihood_value, generation):
        super().__init__()
        self._loglikelihood_value = loglikelihood_value
        self._generation = generation
        self.calls = {
            "loglikelihood": 0,
            "loglikelihood_rolling": 0,
            "generate_until": 0,
        }

    def loglikelihood(self, requests):
        self.calls["loglikelihood"] += len(requests)
        return [self._loglikelihood_value] * len(requests)

    def loglikelihood_rolling(self, requests):
        self.calls["loglikelihood_rolling"] += len(requests)
        return [self._loglikelihood_value[0]] * len(requests)

    def generate_until(self, requests):
        self.calls["generate_until"] += len(requests)
        return [self._generation] * len(requests)


def loglikelihood_request():
    return Instance(
        request_type="loglikelihood",
        doc={},
        arguments=("The capital of France is", " Paris"),
        idx=0,
    )


def generate_request():
    return Instance(
        request_type="generate_until",
        doc={},
        arguments=("The capital of France is", {"until": ["\n"]}),
        idx=0,
    )


def test_a_second_model_does_not_get_the_first_model_s_cached_responses(tmp_path):
    """The defect in #4063: one db, two models, and the second is never run."""
    db = str(tmp_path / "responses_rank0.db")

    first = StubLM((-8.25, False), " i do not know")
    cached_first = CachingLM(first, db, model_identity="model-a::pretrained=a")
    assert cached_first.loglikelihood([loglikelihood_request()]) == [(-8.25, False)]
    assert cached_first.generate_until([generate_request()]) == [" i do not know"]

    second = StubLM((-0.10, True), " Paris")
    cached_second = CachingLM(second, db, model_identity="model-b::pretrained=b")

    assert cached_second.loglikelihood([loglikelihood_request()]) == [(-0.10, True)]
    assert cached_second.generate_until([generate_request()]) == [" Paris"]
    assert second.calls == {
        "loglikelihood": 1,
        "loglikelihood_rolling": 0,
        "generate_until": 1,
    }


def test_the_same_model_still_hits_the_cache(tmp_path):
    """The point of the flag has to keep working."""
    db = str(tmp_path / "responses_rank0.db")
    identity = "model-a::pretrained=a"

    first = StubLM((-8.25, False), " i do not know")
    CachingLM(first, db, model_identity=identity).loglikelihood(
        [loglikelihood_request()]
    )
    assert first.calls["loglikelihood"] == 1

    again = StubLM((-0.10, True), " Paris")
    result = CachingLM(again, db, model_identity=identity).loglikelihood(
        [loglikelihood_request()]
    )

    assert result == [(-8.25, False)]
    assert again.calls["loglikelihood"] == 0


def test_hash_args_separates_identities(tmp_path):
    """The key itself, without the machinery around it."""
    args = ("The capital of France is", " Paris")

    assert hash_args("loglikelihood", args, "model-a") != hash_args(
        "loglikelihood", args, "model-b"
    )
    assert hash_args("loglikelihood", args, "model-a") == hash_args(
        "loglikelihood", args, "model-a"
    )


def test_hash_args_does_not_confuse_the_identity_and_the_request_type(tmp_path):
    """Concatenating the two into one string would make these collide."""
    args = ("ctx", " cont")

    assert hash_args("likelihood", args, "modela") != hash_args(
        "ikelihood", args, "modelal"
    )
