import logging
import time

import requests
from requests.exceptions import RequestException
from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from llama_cpp import Llama

logger = logging.getLogger(__name__)

def get_result(logprobs, context_length):
    is_greedy = True
    offsets = logprobs["text_offset"]
    tokens = logprobs["tokens"]
    tokens_logprobs = logprobs["token_logprobs"]

    idx = 0
    while offsets[idx] < context_length:
        idx += 1
    continuation_logprobs = sum(tokens_logprobs[idx:-1])
    for i in range(idx, len(tokens)):
        top_tokens = logprobs["top_logprobs"][i]
        top_token = max(top_tokens.keys(), key=lambda x: top_tokens[x])
        # can be replaced with 
        # top_token = list(top_tokens.keys())[0]
        if top_token != tokens[i]:
            is_greedy = False
            break

    return continuation_logprobs, is_greedy


@register_model("builtin_gguf")
class BUILTIN_GGUFLM(LM):
    def __init__(self, model=None, max_length=2048, **kwargs):
        super().__init__()
        assert model, "must pass `model` to use MY_GGUF LM!"
        self.model = Llama(
            model_path=model,
            n_gpu_layers=-1, # use GPU acceleration 
            # seed=1337, # set a random seed
            n_ctx=2048,
            logits_all=True,
            verbose=False
        )
        self.logprobs = 1
        self.temperature = 0.0
        self.max_length = max_length

    def gguf_completion(
        self, context, continuation=None, stop=None, **kwargs
    ):
        try:
            prompt = context
            logprobs = self.logprobs
            temperature = self.temperature
            max_tokens = 16
            echo = False

            if continuation:
                prompt += continuation
                max_tokens = 1
                echo = True

            if stop is None:
                stop = []

            response = self.model(
                prompt = prompt, 
                max_tokens = max_tokens, 
                temperature = temperature, 
                logprobs = logprobs,
                echo = echo,
                stop = stop,
            )

            return response
        except ValueError as v:
            logger.error(f"The requested tokens exceed the context window: {v}")
        except RuntimeError as r:
            logger.error(f"the prompt fails to tokenize or the model fails to evaluate the prompt: {r}")

    def loglikelihood(self, requests):
        if not requests:
            return []
        res = []
        for context, continuation in tqdm([req.args for req in requests]):
            response = self.gguf_completion(context=context, continuation=continuation)
            if response and "choices" in response and response["choices"]:
                choice = response["choices"][0]
                logprobs = choice.get("logprobs")
                if (
                    logprobs
                    and "token_logprobs" in logprobs
                    and logprobs["token_logprobs"]
                ):
                    logprob, is_greedy = get_result(logprobs, len(context))
                    res.append((logprob, is_greedy))
                else:
                    logger.warning(
                        "Invalid logprobs data. Expected 'logprobs' to contain 'token_logprobs' list."
                    )
            else:
                logger.error(
                    f"Invalid response for loglikelihood. Response: {response}"
                )
                assert False
        return res

    def generate_until(self, requests):
        if not requests:
            return []

        res = []
        for request in tqdm([req.args for req in requests]):
            inp = request[0]
            request_args = request[1]
            until = request_args.get("until", ["</s>"])
            response = self.gguf_completion(context=inp, stop=until)
            if response and "choices" in response and response["choices"]:
                choice = response["choices"][0]
                if "text" in choice:
                    generated_text = choice["text"].strip()
                    res.append(generated_text)
                else:
                    logger.error(
                        f"Invalid response for greedy_until. Response: {response}"
                    )
                    res.append(None)  # Add default value in case of error
            else:
                logger.error(f"Invalid response for greedy_until. Response: {response}")
                res.append(None)  # Add default value in case of error
        return res

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError(
            "loglikelihood_rolling not yet supported for MY_GGUF models"
        )
