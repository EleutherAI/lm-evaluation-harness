import requests
import logging
from lm_eval.base import BaseLM
from tqdm import tqdm
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)

def ggml_completion(base_url, **kwargs):
    try:
        response = requests.post(f"{base_url}/v1/completions", json=kwargs)
        response.raise_for_status()
        return response.json()
    except RequestException as e:
        print(f"RequestException: {e}")
        return None

class GGMLLM(BaseLM):
    def __init__(self, base_url, truncate=False):
        super().__init__()
        self.base_url = base_url
        self.truncate = truncate

    def loglikelihood(self, requests):
        res = []
        for context, continuation in tqdm(requests):
            response = ggml_completion(self.base_url, context=context, continuation=continuation)
            if response and "choices" in response and response["choices"]:
                choice = response["choices"][0]
                logprobs = choice.get("logprobs")
                logprob = logprobs["token_logprobs"][0] if logprobs and logprobs["token_logprobs"] else -1.2345
                is_greedy = choice["finish_reason"] == "length"
                res.append((logprob, is_greedy))
            else:
                logger.error(f"Invalid response for loglikelihood. Response: {response}")
                assert False
        return res

    def greedy_until(self, requests):
        if not requests:
            return []

        res = []
        for request in tqdm(requests):
            inp = request[0]
            request_args = request[1]
            until = request_args["until"]
            response = ggml_completion(self.base_url, context=inp, stop=until)
            if response and "text" in response:
                generated_text = response["text"].strip()
                res.append(generated_text)
            else:
                logger.error(f"Invalid response for greedy_until. Response: {response}")
                continue
        return res
    
    def _model_call(self, inps):
        # Placeholder implementation
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Placeholder implementation
        raise NotImplementedError()

    @property
    def batch_size(self):
        # Placeholder implementation
        raise NotImplementedError()

    @property
    def device(self):
        # Placeholder implementation
        raise NotImplementedError()

    @property
    def eot_token_id(self):
        # Placeholder implementation
        raise NotImplementedError()

    @property
    def max_length(self):
        # Placeholder implementation
        raise NotImplementedError()

    @property
    def max_gen_toks(self):
        # Placeholder implementation
        raise NotImplementedError()

    def tok_encode(self, string: str):
        # Placeholder implementation
        raise NotImplementedError()

    def tok_decode(self, tokens):
        # Placeholder implementation
        raise NotImplementedError()
