from lm_eval.api.instance import Instance
from lm_eval.api.model import LM, CachingLM, hash_args


class ToyLM(LM):
    def __init__(self):
        super().__init__()
        self.generate_calls = 0

    def loglikelihood(self, requests):
        return [(0.0, True) for _ in requests]

    def loglikelihood_rolling(self, requests):
        return [0.0 for _ in requests]

    def generate_until(self, requests):
        results = []
        for request in requests:
            result = f"generated-{self.generate_calls}"
            self.generate_calls += 1
            self.cache_hook.add_partial("generate_until", request.args, result)
            results.append(result)
        return results


def generate_request(gen_kwargs):
    return Instance(
        request_type="generate_until",
        doc={},
        arguments=("prompt", gen_kwargs),
        idx=0,
        metadata=("task", 0, 1),
    )


def test_caching_lm_does_not_write_sampled_generations(tmp_path):
    lm = ToyLM()
    cached_lm = CachingLM(lm, str(tmp_path / "cache.sqlite"))
    request = generate_request({"do_sample": True, "until": ["\n"]})

    assert cached_lm.generate_until([request]) == ["generated-0"]
    assert hash_args("generate_until", request.args) not in cached_lm.dbdict


def test_caching_lm_reads_deterministic_generations(tmp_path):
    lm = ToyLM()
    cached_lm = CachingLM(lm, str(tmp_path / "cache.sqlite"))
    request = generate_request({"until": ["\n"]})

    assert cached_lm.generate_until([request]) == ["generated-0"]
    assert cached_lm.generate_until([request]) == ["generated-0"]
    assert lm.generate_calls == 1
