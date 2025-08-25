import logging
import time

import requests
from requests.exceptions import RequestException
from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


logger = logging.getLogger(__name__)

def get_result(logprobs):
    is_greedy = True
    content_logprobs = logprobs["content"]
    continuation_logprobs = sum(item["logprob"] for item in content_logprobs)

    for item in content_logprobs:
        top_logprobs_list = item["top_logprobs"]
        if top_logprobs_list:
            top_item = max(top_logprobs_list, key=lambda x: x["logprob"])
            if top_item["token"] != item["token"]:
                is_greedy = False
                break

    return continuation_logprobs, is_greedy


@register_model("gguf", "ggml")
class GGUFLM(LM):
    def __init__(self, base_url=None, max_length=2048, **kwargs):
        super().__init__()
        self.base_url = base_url
        assert self.base_url, "must pass `base_url` to use GGUF LM!"
        self.logprobs = 10
        self.temperature = 0.0
        self.max_length = max_length

    def gguf_completion(
        self, context, continuation=None, stop=None, retries=3, delay=5, **kwargs
    ):
        for _ in range(retries):
            try:
                prompt = context
                request = {
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        },
                    ],
                    "logprobs": True,
                    "top_logprobs": self.logprobs,
                    "temperature": self.temperature,
                }
                if continuation:
                    prompt += continuation
                    request.update({"messages": [
                                   {"role": "user", "content": prompt}], "max_completion_tokens": 1, "max_tokens": 1})

                if stop is not None:
                    request["stop"] = stop
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions", json=request
                )
                response.raise_for_status()
                return response.json()
            except RequestException as e:
                logger.error(f"RequestException: {e}")
                time.sleep(delay)  # wait before retrying
        else:
            raise RuntimeError(
                f"Failed to get a valid response after {retries} retries."
            )

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        if not requests:
            return []
        res = []
        for context, continuation in tqdm(
            [req.args for req in requests], disable=disable_tqdm
        ):

            response = self.gguf_completion(
                context=context, continuation=continuation)
            if response and "choices" in response and response["choices"]:
                choice = response["choices"][0]
                logprobs = choice.get("logprobs")
                if (
                    logprobs
                    and "content" in logprobs
                    and logprobs["content"]
                ):
                    logprob, is_greedy = get_result(logprobs)
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

    def generate_until(self, requests, disable_tqdm: bool = False):
        if not requests:
            return []

        res = []
        for request in tqdm([req.args for req in requests], disable=disable_tqdm):
            inp = request[0]
            request_args = request[1]
            until = request_args.get("until", ["</s>"])
            response = self.gguf_completion(context=inp, stop=until)
            if response and "choices" in response and response["choices"]:
                choice = response["choices"][0]
                if "message" in choice and "content" in choice["message"] and choice["message"]["content"]:
                    generated_text = choice["message"]["content"].strip()
                    res.append(generated_text)
                else:
                    logger.error(
                        f"Invalid response for greedy_until. Response: {response}"
                    )
                    res.append(None)  # Add default value in case of error
            else:
                logger.error(
                    f"Invalid response for greedy_until. Response: {response}")
                res.append(None)  # Add default value in case of error
        return res

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError(
            "loglikelihood_rolling not yet supported for GGUF models"
        )