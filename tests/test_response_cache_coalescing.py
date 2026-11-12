from types import SimpleNamespace

from lm_eval.api.model import CachingLM, LM


class _CountingLM(LM):
    def __init__(self):
        super().__init__()
        self.scores = 0
        self.generations = 0

    def loglikelihood(self, requests):
        start = self.scores
        self.scores += len(requests)
        return [(-float(start + index + 1), False) for index in range(len(requests))]

    def loglikelihood_rolling(self, requests):
        return [-1.0] * len(requests)

    def generate_until(self, requests):
        start = self.generations
        self.generations += len(requests)
        return [f"generation-{start + index + 1}" for index in range(len(requests))]


def test_duplicate_cache_misses_share_one_model_call(tmp_path):
    model = _CountingLM()
    cached = CachingLM(model, str(tmp_path / "responses.db"))
    try:
        request = lambda text: SimpleNamespace(args=(text, " continuation"))
        # First populate one ordinary cache hit.
        assert cached.loglikelihood([request("cached")]) == [(-1.0, False)]

        # The duplicated missing request is interleaved with a cache hit
        # and another distinct miss. Output positions must stay intact.
        batch = [
            request("new"),
            request("cached"),
            request("new"),
            request("other"),
            request("new"),
        ]
        expected = [
            (-2.0, False),
            (-1.0, False),
            (-2.0, False),
            (-3.0, False),
            (-2.0, False),
        ]
        assert cached.loglikelihood(batch) == expected
        assert model.scores == 3

        assert cached.loglikelihood(batch) == expected
        assert model.scores == 3
    finally:
        cached.dbdict.close()


def test_deterministic_generations_share_uncached_requests(tmp_path):
    model = _CountingLM()
    cached = CachingLM(model, str(tmp_path / "responses.db"))
    try:
        request = lambda: SimpleNamespace(args=("prompt", {"until": ["\\n"]}))
        assert cached.generate_until([request(), request()]) == [
            "generation-1",
            "generation-1",
        ]
        assert model.generations == 1
    finally:
        cached.dbdict.close()


def test_sampled_generations_remain_independent(tmp_path):
    model = _CountingLM()
    cached = CachingLM(model, str(tmp_path / "responses.db"))
    try:
        request = lambda: SimpleNamespace(args=("prompt", {"do_sample": True}))
        assert cached.generate_until([request(), request()]) == [
            "generation-1",
            "generation-2",
        ]
        assert model.generations == 2
    finally:
        cached.dbdict.close()
