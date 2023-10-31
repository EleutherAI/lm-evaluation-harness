import requests
import logging
import time
from tqdm import tqdm
from requests.exceptions import RequestException
import transformers
from lm_eval.utils import Reorderer
from lm_eval.base import BaseLM

logger = logging.getLogger(__name__)


def get_result(logprobs, context_lenght):
    is_greedy = True
    offsets = logprobs['text_offset']
    tokens = logprobs['tokens']
    tokens_logprobs = logprobs['token_logprobs']

    idx = 0
    while offsets[idx] < context_lenght:
        idx += 1
    continuation_logprobs = sum(tokens_logprobs[idx:-1])
    for i in range(idx, len(tokens)):
        token = tokens[i]
        top_tokens = logprobs["top_logprobs"][i]
        top_token = max(top_tokens.keys(), key=lambda x: top_tokens[x])
        if top_token != token:
            is_greedy = False
            break

    return continuation_logprobs, is_greedy


class GGMLLM(BaseLM):
    def __init__(self, base_url, truncate=False):
        super().__init__()
        self.base_url = base_url
        self.truncate = truncate
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
        self.logpobs = 10
        self.temperature = 0.0
        self.max_length = 1024
        self.vocab_size = self.tokenizer.vocab_size

    def ggml_completion(self, context, continuation=None, stop=None, retries=3, delay=5, **kwargs):
        for _ in range(retries):
            try:
                prompt = context
                request = {'prompt': prompt, 'logprobs': self.logpobs,
                           'temperature': self.temperature}
                if continuation:
                    prompt += continuation
                    request.update({'prompt': prompt, 'max_tokens': 1, 'echo': True})
                if stop is not None:
                    request['stop'] = stop
                response = requests.post(f"{self.base_url}/v1/completions", json=request)
                response.raise_for_status()
                return response.json()
            except RequestException as e:
                logger.error(f"RequestException: {e}")
                time.sleep(delay)  # wait before retrying
        else:
            raise Exception(f"Failed to get a valid response after {retries} retries.")

    def loglikelihood(self, requests):
        if not requests:
            return []
        res = []
        for context, continuation in tqdm(requests):
            response = self.ggml_completion(context=context, continuation=continuation)
            if response and "choices" in response and response["choices"]:
                choice = response["choices"][0]
                logprobs = choice.get("logprobs")
                if logprobs and "token_logprobs" in logprobs and logprobs["token_logprobs"]:
                    logprob, is_greedy = get_result(logprobs, len(context))
                    res.append((logprob, is_greedy))
                else:
                    logger.warning("Invalid logprobs data. Expected 'logprobs' to contain 'token_logprobs' list.")
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
            response = self.ggml_completion(context=inp, stop=until)
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
        return res

    def loglikelihood_rolling(self, requests):
        results = []

        for request in requests:
            logprobs = []
            for i in range(0, len(request), self.max_length):
                chunk = request[i:i + self.max_length]
                chunk_loglikelihood = self.loglikelihood([(chunk, request[i + 1:i + self.max_length + 1])])
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

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

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

    def max_length(self):
        return self.max_length

    @property
    def max_gen_toks(self):
        # Placeholder implementation
        raise NotImplementedError()
