import os
import unittest
from unittest.mock import patch

from lm_eval.api.instance import Instance


API_KEY_ENV = {"TEXTSYNTH_API_SECRET_KEY": "test-secret"}


class FakeResponse:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


def loglikelihood_instances(args_list):
    return [
        Instance(
            request_type="loglikelihood",
            doc={"context": ctx, "continuation": cont},
            arguments=(ctx, cont),
            idx=i,
        )
        for i, (ctx, cont) in enumerate(args_list)
    ]


def generate_until_instances(args_list):
    return [
        Instance(
            request_type="generate_until",
            doc={"input": inp},
            arguments=(inp, gen_kwargs),
            idx=i,
        )
        for i, (inp, gen_kwargs) in enumerate(args_list)
    ]


@patch.dict(os.environ, API_KEY_ENV)
class TextSynthLMTest(unittest.TestCase):
    def _make_lm(self, engine="gptj_6B"):
        from lm_eval.models.textsynth import TextSynthLM

        return TextSynthLM(engine=engine)

    def test_loglikelihood_parses_response_and_builds_request(self):
        # The evaluator passes `Instance` objects (see evaluator.py), so the
        # backend must read `.args`. Iterating the instances directly would
        # raise, so this also guards against that regression.
        calls = []

        def fake_post(url=None, headers=None, json=None, **kwargs):
            calls.append({"url": url, "headers": headers, "json": json})
            return FakeResponse({"logprob": -1.5, "is_greedy": True})

        with patch("lm_eval.models.textsynth._requests.post", side_effect=fake_post):
            lm = self._make_lm()
            res = lm.loglikelihood(loglikelihood_instances([("ctx", "cont")]))

        self.assertEqual(res, [(-1.5, True)])
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["url"].endswith("/v1/engines/gptj_6B/logprob"))
        self.assertEqual(calls[0]["json"], {"context": "ctx", "continuation": "cont"})
        self.assertEqual(calls[0]["headers"], {"Authorization": "Bearer test-secret"})

    def test_loglikelihood_missing_logprob_raises(self):
        def fake_post(**kwargs):
            return FakeResponse({"error": "no logprob here"})

        with patch("lm_eval.models.textsynth._requests.post", side_effect=fake_post):
            lm = self._make_lm()
            with self.assertRaises(AssertionError):
                lm.loglikelihood(loglikelihood_instances([("ctx", "cont")]))

    def test_loglikelihood_rolling_not_implemented(self):
        with patch("lm_eval.models.textsynth._requests.post"):
            lm = self._make_lm()
        with self.assertRaises(NotImplementedError):
            lm.loglikelihood_rolling(loglikelihood_instances([("ctx", "cont")]))

    def test_generate_until_parses_text_and_builds_request(self):
        calls = []

        def fake_post(url=None, headers=None, json=None, **kwargs):
            calls.append({"url": url, "headers": headers, "json": json})
            return FakeResponse({"text": "generated output"})

        with patch("lm_eval.models.textsynth._requests.post", side_effect=fake_post):
            lm = self._make_lm()
            res = lm.generate_until(
                generate_until_instances([("hello", {"until": ["\n", "END"]})])
            )

        self.assertEqual(res, ["generated output"])
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["url"].endswith("/v1/engines/gptj_6B/completions"))
        self.assertEqual(
            calls[0]["json"],
            {
                "prompt": "hello",
                "max_tokens": 256,
                "top_k": 1,
                "stop": ["\n", "END"],
            },
        )

    def test_generate_until_empty_requests_returns_empty(self):
        with patch("lm_eval.models.textsynth._requests.post") as mock_post:
            lm = self._make_lm()
            self.assertEqual(lm.generate_until([]), [])
            mock_post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
