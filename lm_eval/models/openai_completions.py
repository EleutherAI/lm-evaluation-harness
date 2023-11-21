import os
import time
from typing import List, Tuple
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
        if not requests:
            return []
        res = []
        requests = [req.args for req in requests]

        def _collate(x):
            toks = self.tok_encode(x[0])
            return len(toks), x[0]

        re_ord = utils.Reorderer(requests, _collate)

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

        # todo: more intelligent batching for heterogeneous `until`
        for chunk, request_args in tqdm(
                list(sameuntil_chunks(re_ord.get_reordered(), self.REQ_CHUNK_SIZE))
        ):
            inps = []
            for context, _ in chunk:
                # context_enc = self.tok_encode(context)
                # inp = context_enc[-(self.max_length - self.max_gen_toks):]
                inps.append({"role": "user", "content": context})

            # until = request_args.get("until", ["<|endoftext|>"])
            until = request_args.get("until", None)

            response = oa_chat_completion(
                messages=inps,
                model=self.model,
                frequency_penalty=self.frequency_penalty,
                # logit_bias=self.logit_bias,
                max_tokens=self.max_gen_toks,
                n=self.n,
                presence_penalty=self.presence_penalty,
                temperature=self.temperature,
                top_p=self.top_p,
                # stop=until,
            )

            for resp, (context, args_) in zip(response.choices, chunk):
                print(resp)
                import sys; sys.exit()

                s = resp["text"]

                # until_ = args_.get("until", ["<|endoftext|>"])
                until_ = args_.get("until", "null")

                for term in until_:
                    if len(term) > 0:
                        s = s.split(term)[0]

                # partial caching
                self.cache_hook.add_partial(
                    "generate_until", (context, {"until": until_}), s
                )

                res.append(s)
        return re_ord.get_original(res)

    def loglikelihood(self, requests):
        raise NotImplementedError("No support for logits.")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("No support for logits.")
