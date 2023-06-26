import requests
import json
from tqdm import tqdm
from requests.exceptions import RequestException
import time

def llama_completion(base_url, prompt, **kwargs):
    try:
        response = requests.post(f"{base_url}/v1/completions", json=kwargs)
        response.raise_for_status()
        return response.json()
    except RequestException as e:
        print(f"RequestException: {e}")
        return None

class LlamaLM(BaseLM):
    def __init__(self, base_url, truncate=False):
        super().__init__()
        self.base_url = base_url
        self.truncate = truncate

    def loglikelihood(self, requests):
        res = []
        for context, continuation in tqdm(requests):
            response = llama_completion(self.base_url, context, continuation=continuation)
            if response and "logprob" in response:
                logprob = response["logprob"]
                is_greedy = response["is_greedy"]
                res.append((logprob, is_greedy))
            else:
                logger.error("Invalid response for loglikelihood")
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
            response = llama_completion(self.base_url, inp, stop=until)
            if response and "text" in response:
                s = response["text"]
                res.append(s)
            else:
                logger.error("Invalid response for greedy_until")
                assert False
        return res
