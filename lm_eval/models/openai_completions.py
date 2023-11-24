import os
import time
from typing import List, Tuple

import copy
from collections import defaultdict
from tqdm import tqdm

from lm_eval import utils
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from openai import OpenAI

client = OpenAI()


def oa_chat_completion(**kwargs):
    """Query OpenAI API for chat completion.

    Retry with back-off until they respond
    """
    try:
        import openai, tiktoken  # noqa: E401
    except ModuleNotFoundError:
        raise Exception(
            "attempted to use 'openai' LM type, but package `openai` or `tiktoken` are not installed. \
please install these via `pip install lm-eval[openai]` or `pip install -e .[openai]`",
        )

    backoff_time = 3
    while True:
        try:
            return client.chat.completions.create(**kwargs)
        except openai.OpenAIError:
            import traceback

            traceback.print_exc()
            time.sleep(backoff_time)
            backoff_time *= 1.5


@register_model("openai-chat-completions")
class OpenaiChatCompletionsLM(LM):
    REQ_CHUNK_SIZE = 20

    def __init__(
        self, model: str = "gpt-3.5-turbo", truncate: bool = False, batch_size: int = 1
    ) -> None:
        """

        :param model: str
            OpenAI API model (e.g. gpt-3.5-turbo)
        :param truncate: bool
            Truncate input if too long (if False and input is too long, throw error)
        """
        super().__init__()
        try:
            import openai, tiktoken  # noqa: E401
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'openai' LM type, but package `openai` or `tiktoken` are not installed. \
    please install these via `pip install lm-eval[openai]` or `pip install -e .[openai]`",
            )
        self.model = model
        self.frequency_penalty = 0
        self.logit_bias = None
        self.n = 1
        self.presence_penalty = 0
        self.temperature = 1
        self.top_p = 1
        self.tokenizer = tiktoken.encoding_for_model(self.model)
        self.vocab_size = self.tokenizer.n_vocab
        self.truncate = truncate
        self.end_of_text_token_id = self.tokenizer.eot_token

        # Read from environment variable OPENAI_API_SECRET_KEY

    @property
    def eot_token_id(self):
        return self.end_of_text_token_id

    @property
    def max_length(self) -> int:
        # Note: the OpenAI API supports up to 2049 tokens, with the first token being the first input token
        return 2048

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def tok_encode(self, string: str) -> List[int]:
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    def _encode_pair(
        self, context: str, continuation: str
    ) -> Tuple[List[int], List[int]]:
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole_enc = self.tok_encode(context + continuation)
        context_enc = self.tok_encode(context)
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc

    def generate_until(self, requests) -> List[str]:
        res = defaultdict(list)
        re_ords = {}

        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        grouper = utils.Grouper(requests, lambda x: str(x.args[1]))
        for key, reqs in grouper.get_grouped().items():
            # within each set of reqs for given kwargs, we reorder by token length, descending.
            re_ords[key] = utils.Reorderer([req.args for req in reqs], _collate)

        def sameuntil_chunks(xs, size):
            ret = []
            lastuntil = xs[0][1]
            for x in xs:
                if len(ret) >= size or x[1] != lastuntil:
                    yield ret, lastuntil
                    ret = []
                    lastuntil = x[1]
                ret.append(x)

            if ret:
                yield ret, lastuntil

        pbar = tqdm(total=len(requests), disable=(self.rank != 0))
        for key, re_ord in re_ords.items():
            chunks = utils.chunks(re_ord.get_reordered(), n=self.REQ_CHUNK_SIZE)
            for chunk in chunks:
                contexts, all_gen_kwargs = zip(*chunk)
                inps = [{"role": "user", "content": context} for context in contexts]

            gen_kwargs = all_gen_kwargs[0]
            until = None
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                if "until" in kwargs.keys():
                    until = kwargs.pop("until")
                    if isinstance(until, str):
                        until = [kwargs]
                    elif not isinstance(until, list):
                        raise ValueError(
                            f"Expected `kwargs['until']` to be of type Union[str,list] but got {until}"
                        )
            else:
                raise ValueError(
                    f"Expected `kwargs` to be of type `dict` but got {kwargs}"
                )

            if "max_gen_toks" in kwargs.keys():
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks

            response = oa_chat_completion(
                messages=inps,
                model=self.model,
                frequency_penalty=self.frequency_penalty,
                # logit_bias=self.logit_bias,
                max_tokens=max_gen_toks,
                n=self.n,
                presence_penalty=self.presence_penalty,
                temperature=self.temperature,
                top_p=self.top_p,
            )

            for resp, (context, args_) in zip(response.choices, chunk):
                s = resp.message.content

                if until is not None:
                    for term in until:
                        if len(term) > 0:
                            s = s.split(term)[0]

                res[key].append(s)

                self.cache_hook.add_partial(
                    "generate_until", (context, {"until": until}), s
                )
                pbar.update(1)

            res[key] = re_ord.get_original(res[key])

        pbar.close()

        return grouper.get_original(res)

    def loglikelihood(self, requests):
        raise NotImplementedError("No support for logits.")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("No support for logits.")
