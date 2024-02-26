import hashlib
import json
import os
import pickle
import unittest
from unittest.mock import patch

from lm_eval.api.instance import Instance
from lm_eval.models.gguf import GGUFLM


base_url = "https://matthoffner-ggml-llm-api.hf.space"


def gguf_completion_mock(base_url=None, **kwargs):
    # Generate a hash from the parameters
    hash_kwargs = {"base_url": base_url, **kwargs}
    hash = hashlib.sha256(
        json.dumps(hash_kwargs, sort_keys=True).encode("utf-8")
    ).hexdigest()

    fname = f"./tests/testdata/gguf_test_{hash}.pkl"

    if os.path.exists(fname):
        with open(fname, "rb") as fh:
            return pickle.load(fh)
    else:
        print("The file does not exist, attempting to write...")
        if "stop" in kwargs:
            result = {
                "choices": [
                    {
                        "text": f"generated text until {kwargs['stop']}",
                        "logprobs": {"token_logprobs": [-1.2345], "text_offset": 0},
                        "finish_reason": "length",
                    }
                ]
            }
        else:
            # generated with # curl -X 'POST'   'http://localhost:8000/v1/completions'   -H 'accept: application/json'   -H 'Content-Type: application/json'   -d '{"prompt": "string", "logprobs": 10, "temperature": 0.0, "max_tokens": 1, "echo": true}'
            result = {
                "id": "cmpl-4023976b-bc6a-43b0-a5a9-629f4216c7f3",
                "object": "text_completion",
                "created": 1700511361,
                "model": "../llama-2-7b.Q8_0.gguf",
                "choices": [
                    {
                        "text": "string(",
                        "index": 0,
                        "logprobs": {
                            "text_offset": [0, 7],
                            "token_logprobs": [None, -1.033263319857306],
                            "tokens": [" string", "("],
                            "top_logprobs": [
                                None,
                                {
                                    "(": -1.033263319857306,
                                    "[]": -2.6530743779017394,
                                    ".": -3.0377145947291324,
                                    "\n": -3.0399156750513976,
                                    "_": -3.510376089937872,
                                    " =": -3.6957918347193663,
                                    ",": -3.9309459866358702,
                                    " of": -4.2834550083949035,
                                    '("': -4.322762841112799,
                                    "()": -4.426229113466925,
                                },
                            ],
                        },
                        "finish_reason": "length",
                    }
                ],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "total_tokens": 3,
                },
            }

        try:
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            print("Writing file at", fname)
            with open(fname, "wb") as fh:
                pickle.dump(result, fh)
            print("File written successfully")
        except Exception as e:
            print("File writing failed:", e)

        return result


class GGUFLMTest(unittest.TestCase):
    @patch(
        "lm_eval.models.gguf.GGUFLM.gguf_completion", side_effect=gguf_completion_mock
    )
    def test_loglikelihood(self, gguf_completion_mock):
        lm = GGUFLM(base_url)

        # Test loglikelihood
        requests = [
            Instance(
                request_type="loglikelihood",
                doc=args,
                arguments=args,
                idx=i,
            )
            for i, args in enumerate([("str", "ing"), ("str", "ing")])
        ]
        res = lm.loglikelihood(requests)

        # Assert the loglikelihood response is correct
        expected_res = [(logprob, True) for logprob in [0, 0]]
        self.assertEqual(res, expected_res)

    @patch(
        "lm_eval.models.gguf.GGUFLM.gguf_completion", side_effect=gguf_completion_mock
    )
    def test_generate_until(self, gguf_completion_mock):
        lm = GGUFLM(base_url)

        # Test generate_until
        requests = [
            Instance(
                request_type="generate_until",
                doc={"input": doc},
                arguments=(doc, {"until": stop}),
                idx=i,
            )
            for i, (doc, stop) in enumerate([("input1", "stop1"), ("input2", "stop2")])
        ]

        res = lm.generate_until(requests)

        # Assert the generate_until response is correct
        expected_res = ["generated text until stop1", "generated text until stop2"]
        self.assertEqual(res, expected_res)

    # @patch('lm_eval.models.gguf.GGUFLM.gguf_completion', side_effect=gguf_completion_mock)
    # def test_loglikelihood_rolling(self, gguf_completion_mock):
    #     lm = GGUFLM(base_url)

    #     # Test loglikelihood_rolling
    #     requests = ["input1", "input2"]
    #     res = lm.loglikelihood_rolling(requests)

    #     # Assert the loglikelihood_rolling response is correct
    #     expected_res = [(-1.2345, True), (-1.2345, True)]
    #     self.assertEqual(res, expected_res)


if __name__ == "__main__":
    unittest.main()
