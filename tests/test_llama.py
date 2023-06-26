import unittest
from unittest.mock import MagicMock
from lm_eval.models.llama import LlamaLM

class LlamaLMTest(unittest.TestCase):
    def test_loglikelihood(self):
        base_url = "https://matthoffner-ggml-llm-api.hf.space"
        lm = LlamaLM(base_url)

        # Create a MagicMock object to mock llama_completion
        llama_completion_mock = MagicMock()

        # Set the return value for the mocked function
        llama_completion_mock.return_value = {
            "logprob": -1.2345,
            "is_greedy": True
        }

        # Patch the llama_completion function with the mocked function
        lm.llama_completion = llama_completion_mock

        # Test loglikelihood
        requests = [("context1", "continuation1"), ("context2", "continuation2")]
        res = lm.loglikelihood(requests)

        # Assert the loglikelihood response is correct
        expected_res = [(-1.2345, True), (-1.2345, True)]
        self.assertEqual(res, expected_res)

    def test_greedy_until(self):
        base_url = "https://matthoffner-ggml-llm-api.hf.space"
        lm = LlamaLM(base_url)

        # Define the llama_completion method with the desired behavior
        def llama_completion_mock(url, context, stop=None):
            if stop is not None:
                return {"text": f"generated_text{stop[-1]}"}
            return {"text": "generated_text"}

        # Set the llama_completion method to the defined mock
        lm.llama_completion = llama_completion_mock

        # Test greedy_until
        requests = [("input1", {"until": "stop1"}), ("input2", {"until": "stop2"})]
        res = lm.greedy_until(requests)

        # Assert the greedy_until response is correct
        expected_res = ["generated_text1", "generated_text2"]
        self.assertEqual(res, expected_res)




if __name__ == "__main__":
    unittest.main()
