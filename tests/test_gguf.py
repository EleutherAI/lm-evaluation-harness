import unittest
from unittest.mock import patch
import hashlib
import json
import os
import pickle
from lm_eval.models.gguf import GGUFLM

base_url = "https://matthoffner-ggml-llm-api.hf.space"

def gguf_completion_mock(base_url, **kwargs):
    # Generate a hash from the parameters
    hash_kwargs = {'base_url': base_url, **kwargs}
    hash = hashlib.sha256(json.dumps(hash_kwargs, sort_keys=True).encode('utf-8')).hexdigest()

    fname = f"./tests/testdata/ggml_test_{hash}.pkl"

    if os.path.exists(fname):
        with open(fname, "rb") as fh:
            return pickle.load(fh)
    else:
        print("The file does not exist, attempting to write...")  
        if 'stop' in kwargs:
            result = {"choices": [{"text": f"generated text until {kwargs['stop']}", "logprobs": {"token_logprobs": [-1.2345]}, "finish_reason": "length"}]}
        else:
            result = {"choices": [{"logprobs": {"token_logprobs": [-1.2345]}, "finish_reason": "length"}]}

        try:
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            print('Writing file at', fname)
            with open(fname, "wb") as fh:
                pickle.dump(result, fh)
            print('File written successfully')
        except Exception as e:
            print('File writing failed:', e)

        return result


class GGUFLMTest(unittest.TestCase):
    @patch('lm_eval.models.gguf.GGUFLM.gguf_completion', side_effect=gguf_completion_mock)
    def test_loglikelihood(self, gguf_completion_mock):
        lm = GGUFLM(base_url)

        # Test loglikelihood
        requests = [("context1", "continuation1"), ("context2", "continuation2")]
        res = lm.loglikelihood(requests)

        # Assert the loglikelihood response is correct
        expected_res = [(logprob, True) for logprob in [-1.2345, -1.2345]]
        self.assertEqual(res, expected_res)

    @patch('lm_eval.models.gguf.GGUFLM.gguf_completion', side_effect=gguf_completion_mock)
    def test_greedy_until(self, gguf_completion_mock):
        lm = GGUFLM(base_url)

        # Test greedy_until
        requests = [("input1", {"until": "stop1"}), ("input2", {"until": "stop2"})]
        res = lm.greedy_until(requests)

        # Assert the greedy_until response is correct
        expected_res = ["generated text until stop1", "generated text until stop2"]
        self.assertEqual(res, expected_res)

    @patch('lm_eval.models.gguf.GGUFLM.gguf_completion', side_effect=gguf_completion_mock)
    def test_loglikelihood_rolling(self, gguf_completion_mock):
        lm = GGUFLM(base_url)

        # Test loglikelihood_rolling
        requests = ["input1", "input2"]
        res = lm.loglikelihood_rolling(requests)

        # Assert the loglikelihood_rolling response is correct
        expected_res = [(-1.2345, True), (-1.2345, True)]
        self.assertEqual(res, expected_res)


if __name__ == "__main__":
    unittest.main()
