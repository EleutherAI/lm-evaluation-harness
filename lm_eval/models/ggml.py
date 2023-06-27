import requests
import logging
import time
from tqdm import tqdm
from requests.exceptions import RequestException

from lm_eval.utils import Reorderer
from lm_eval.base import BaseLM

logger = logging.getLogger(__name__)

def ggml_completion(base_url, retries=3, delay=5, **kwargs):
    for _ in range(retries):
        try:
            response = requests.post(f"{base_url}/v1/completions", json=kwargs)
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            logger.error(f"RequestException: {e}")
            time.sleep(delay)  # wait before retrying
    else:
        raise Exception(f"Failed to get a valid response after {retries} retries. Last exception: {e}")

class GGMLLM(BaseLM):
    def __init__(self, base_url, truncate=False):
        super().__init__()
        self.base_url = base_url
        self.truncate = truncate

    def loglikelihood(self, requests):
        reorderer = Reorderer(requests, len)
        requests = reorderer.get_reordered()

        res = []
        for context, continuation in tqdm(requests):
            response = ggml_completion(self.base_url, context=context, continuation=continuation)
            if response and "choices" in response and response["choices"]:
                choice = response["choices"][0]
                logprobs = choice.get("logprobs")
                try:
                    logprob = logprobs["token_logprobs"][0]
                except TypeError:
                    raise ValueError("Invalid logprobs data. Expected 'logprobs' to contain 'token_logprobs' list.")
                is_greedy = choice["finish_reason"] == "length"
                res.append((logprob, is_greedy))
            else:
                logger.error(f"Invalid response for loglikelihood. Response: {response}")
                assert False
        return reorderer.get_original(res)

    def greedy_until(self, requests):
        if not requests:
            return []

        reorderer = Reorderer(requests, len)
        requests = reorderer.get_reordered()

        res = []
        for request in tqdm(requests):
            inp = request[0]
            request_args = request[1]
            until = request_args["until"]
            response = ggml_completion(self.base_url, context=inp, stop=until)
            print(response);
            if response and "choices" in response and response["choices"]:
                choice = response["choices"][0]
                if "text" in choice:
                    generated_text = choice["text"].strip()
                    res.append(generated_text)
                else:
                    logger.error(f"Invalid response for greedy_until. Response: {response}")
                    res.append(None)  # Add default value in case of error
            else:
                logger.error(f"Invalid response for greedy_until. Response: {response}")
                res.append(None)  # Add default value in case of error
        return reorderer.get_original(res)

    def loglikelihood_rolling(self, requests):
        results = []

        for request in requests:
            logprobs = []
            for i in range(0, len(request), self.max_length):
                chunk = request[i:i+self.max_length]
                chunk_loglikelihood = self.loglikelihood([(chunk, request[i+1:i+self.max_length+1])])
                logprobs.extend(chunk_loglikelihood)
            
            avg_loglikelihood = sum([logprob for logprob, _ in logprobs]) / len(logprobs)
            results.append((avg_loglikelihood, True))

        return results

    
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
        return 1024

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
