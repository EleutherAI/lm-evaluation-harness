""" TextSynth API
Implementation provided by Fabrice Bellard:
    https://github.com/EleutherAI/lm-evaluation-harness/issues/295

In order to use the API, you must have a valid TextSynth account and
enough credits.

Example usage:

    python main.py --model textsynth --model_args engine=gptj_6B --no_cache --tasks piqa

Homepage: https://textsynth.com/index.html
"""
import logging
import os

import requests as _requests
from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import retry_on_specific_exceptions


logger = logging.getLogger(__name__)


def textsynth_completion(**kwargs):
    """Query TextSynth API for completion.
    Retry with back-off until they respond.
    """

    def _exception_callback(e: Exception, sleep_time: float) -> None:
        import traceback

        traceback.print_exc()

    @retry_on_specific_exceptions(
        on_exceptions=[_requests.exceptions.RequestException],
        max_retries=None,  # retry forever, consider changing
        on_exception_callback=_exception_callback,
    )
    def completion():
        return _requests.post(**kwargs)

    return completion()


@register_model("textsynth")
class TextSynthLM(LM):
    def __init__(self, engine, truncate: bool = False, **kwargs) -> None:
        """
        :param engine: str
            TextSynth API engine (e.g. `gptj_6B`)
        :param truncate: bool
            Truncate input if too long (if False and input is too long, throw error)
        """
        super().__init__()

        self.engine = engine
        self.truncate = truncate
        self.api_url = "https://api.textsynth.com"
        # Read from environment variable TEXTSYNTH_API_SECRET_KEY
        self.api_key = os.environ["TEXTSYNTH_API_SECRET_KEY"]

    @property
    def eot_token_id(self):
        # Isn't used because we override loglikelihood, loglikelihood_rolling and generate_until
        raise NotImplementedError()

    @property
    def max_length(self) -> int:
        # NOTE: Turn on truncation to avoid errors on long inputs.
        return 2048

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self):
        # Isn't used because we override loglikelihood, loglikelihood_rolling and generate_until
        raise NotImplementedError()

    @property
    def device(self):
        # Isn't used because we override loglikelihood, loglikelihood_rolling and generate_until
        raise NotImplementedError()

    def tok_encode(self, string: str):
        # Isn't used because we override loglikelihood, loglikelihood_rolling and generate_until
        raise NotImplementedError()

    def tok_decode(self, tokens):
        # Isn't used because we override loglikelihood, loglikelihood_rolling and generate_until
        raise NotImplementedError()

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        res = []
        for context, continuation in tqdm(requests, disable=disable_tqdm):
            response = textsynth_completion(
                url=self.api_url + "/v1/engines/" + self.engine + "/logprob",
                headers={"Authorization": "Bearer " + self.api_key},
                json={"context": context, "continuation": continuation},
            )
            resp = response.json()
            if "logprob" in resp:
                logprob = resp["logprob"]
                is_greedy = resp["is_greedy"]
                res.append((logprob, is_greedy))

                self.cache_hook.add_partial(
                    "loglikelihood", (context, continuation), (logprob, is_greedy)
                )
            else:
                logger.error(
                    f"The following response does not contain `logprobs`. Got:\n{resp}"
                )
                assert False
        return res

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        # TODO: The TextSynth API does not support tokenized inputs so we cannot
        # manually partition long contexts into smaller rolling windows as
        # done for other models derived from `BaseLM`. Override this method
        # with a windowing scheme that works for direct string inputs.
        raise NotImplementedError(
            "`loglikelihood_rolling` is currently not supported due to lack of "
            "input tokenization support from TextSynth."
        )

    def generate_until(self, requests, disable_tqdm: bool = False):
        if not requests:
            return []

        res = []
        for request in tqdm(requests, disable=disable_tqdm):
            inp = request[0]
            request_args = request[1]
            until = request_args["until"]
            response = textsynth_completion(
                url=self.api_url + "/v1/engines/" + self.engine + "/completions",
                headers={"Authorization": "Bearer " + self.api_key},
                json={
                    "prompt": inp,
                    "max_tokens": self.max_gen_toks,
                    "top_k": 1,
                    "stop": until,
                },
            )
            resp = response.json()
            if "text" in resp:
                s = resp["text"]
                res.append(s)

                self.cache_hook.add_partial("generate_until", (inp, request_args), s)
            else:
                logger.error(
                    "The following response does not contain generated `text`. "
                    "Got:\n{resp}"
                )
                assert False
        return res

    def _model_call(self, inps):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override generate_until
        raise NotImplementedError()
