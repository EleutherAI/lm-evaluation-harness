import os
import numpy as np
import transformers
from lm_eval.base import BaseLM
from lm_eval import utils
from tqdm import tqdm
import time


def get_result(response, ctxlen):
    """Process results from OpenAI API response.

    :param response: dict
        OpenAI API Response
    :param ctxlen: int
        Length of context (so we can slice them away and only keep the predictions)
    :return:
        continuation_logprobs: np.array
            Log probabilities of continuation tokens
        is_greedy: bool
            whether argmax matches given continuation exactly
    """
    is_greedy = True
    logprobs = response["logprobs"]["token_logprobs"]
    continuation_logprobs = sum(logprobs[ctxlen:])

    for i in range(ctxlen, len(response["logprobs"]["tokens"])):
        token = response["logprobs"]["tokens"][i]
        top_tokens = response["logprobs"]["top_logprobs"][i]
        top_token = max(top_tokens.keys(), key=lambda x: top_tokens[x])
        if top_token != token:
            is_greedy = False
            break

    return continuation_logprobs, is_greedy


def oa_completion(**kwargs):
    """ Query OpenAI API for completion.

    Retry with back-off until they respond
    """
    import openai
    backoff_time = 3
    while True:
        try:
            return openai.Completion.create(**kwargs)
        except openai.error.OpenAIError:
            import traceback
            traceback.print_exc()
            time.sleep(backoff_time)
            backoff_time *= 1.5


class OpenAICompletionsLM(BaseLM):
    """
    Implements the BaseLM interface for OpenAI's Completions API.
    See: https://beta.openai.com/docs/api-reference/completions
    """

    def __init__(
        self,
        engine: str,
        device=None,
        batch_size: int = 20,
        max_gen_toks: int = 256,
        parallelize=False
    ):
        """

        :param engine: str
            OpenAI API engine (e.g. davinci)
        :param truncate: bool
            Truncate input if too long (if False and input is too long, throw error)
        """
        super().__init__()

        assert device is None, "Cannot specify `device` - GPT-3 is only accessible through the OpenAI API."
        assert parallelize == False, "Cannot specify `parallelize` - GPT-3 is only accessible through the OpenAI API."

        import openai

        self.engine = engine
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')
        # To make the annoying "Using pad_token, but it is not set yet." error go away
        self.tokenizer.pad_token = "<|endoftext|>"
        self.vocab_size = self.tokenizer.vocab_size

        self._max_gen_toks = max_gen_toks
        self._batch_size = batch_size  # todo: adaptive batch size

        # Read from environment variable OPENAI_API_SECRET_KEY
        openai.api_key = os.environ["OPENAI_API_SECRET_KEY"]

    @property
    def eot_token(self):
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        # Note: the OpenAI API supports up to 2049 tokens, with the first token being the first input token
        return 2048

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        res = []

        def _collate(x):
            # this doesn't efficiently handle last-token differences yet, but those are kinda annoying because
            # it's not guaranteed that the 100 or so logprobs we get to see actually contain all the continuations
            # we care about and so we need some kind of backup for when it isn't
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        reord = utils.Reorderer(requests, _collate)

        for chunk in tqdm(list(utils.chunks(reord.get_reordered(), self.batch_size)), disable=disable_tqdm):
            inps = []
            ctxlens = []
            for cache_key, context_enc, continuation_enc in chunk:
                # max_length+1 because the API takes up to 2049 tokens, including the first context token
                inp = (context_enc + continuation_enc)[-(self.max_length+1):]
                # TODO: the logic is much simpler if we just look at the length of continuation tokens
                ctxlen = len(context_enc) - max(0, len(context_enc) +
                                                len(continuation_enc) - (self.max_length+1))

                inps.append(inp)
                ctxlens.append(ctxlen)

            response = self._model_call(inps)

            for resp, ctxlen, (cache_key, context_enc, continuation_enc) in zip(response.choices, ctxlens, chunk):
                answer = get_result(resp, ctxlen)

                res.append(answer)

                # partial caching
                if cache_key is not None:
                    self.cache_hook.add_partial(
                        "loglikelihood", cache_key, answer)

        return reord.get_original(res)

    def greedy_until(self, requests):
        if not requests:
            return []
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return len(toks), x[0]

        reord = utils.Reorderer(requests, _collate)

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
        for chunk, request_args in tqdm(list(sameuntil_chunks(reord.get_reordered(), self.batch_size))):
            stopping_criteria = request_args["stopping_criteria"]
            max_generation_length = request_args["max_generation_length"]
            num_fewshot = request_args["num_fewshot"]

            assert isinstance(stopping_criteria,
                              str) or stopping_criteria is None
            assert (
                isinstance(max_generation_length,
                           int) or max_generation_length is None
            )
            assert isinstance(num_fewshot, int) or num_fewshot is None

            # TODO(jon-tow): This is most likely useless b/c `stopping_criteria` is
            # never `None`; see `base.py` `PromptSourceTask.construct_requests`.
            if stopping_criteria is None:
                until = [self.eot_token]
            else:
                until = [stopping_criteria]

            inps = []
            for context, _ in chunk:
                context_enc = self.tok_encode(context)
                inp = context_enc[-(self.max_length - self.max_gen_toks):]
                inps.append(inp)

            if max_generation_length is None:
                max_length = self.max_gen_toks
            else:
                max_length = max_generation_length

            response = self._model_generate(
                context=inps,
                max_length=max_length,
                # NOTE: We do not need to tokenize the stopping criteria with the OpenAI API
                # so just pass in the list of stopping tokens.
                stopping_criteria_ids=until,
                num_fewshot=num_fewshot,
            )

            # Iterate thru the per-request responses.
            for resp, (context, _request_args) in zip(response.choices, chunk):
                sentence = resp['text']

                _stopping_criteria = _request_args["stopping_criteria"]
                _until = [self.eot_token] if _stopping_criteria is None else [
                    _stopping_criteria]

                for term in _until:
                    sentence = sentence.split(term)[0]

                # partial caching
                self.cache_hook.add_partial(
                    "greedy_until", (context, _until), sentence)

                res.append(sentence)

        return reord.get_original(res)

    def _model_call(self, inps):
        return oa_completion(
            engine=self.engine,
            prompt=inps,
            echo=True,
            max_tokens=0,
            temperature=0.,
            logprobs=5,
        )

    def _model_generate(self, context, max_length, stopping_criteria_ids, num_fewshot):
        """ 
        NOTE: Rename `stopping_criteria_ids` to `stopping_criteria`b/c We do not 
        need to tokenize the stopping sequences in the OpenAI API
        """

        # NOTE: We don't need to tokenize the stopping sequences into ids when
        # using the OpenAI API.
        stopping_criteria = stopping_criteria_ids

        # NOTE: We don't need to add context size b/c OpenAI completion only
        # expects the max generation count portion.

        if num_fewshot == 0:
            generations = oa_completion(
                engine=self.engine,
                prompt=context,
                max_tokens=max_length,
                temperature=0.,
                logprobs=5,
                stop=[self.eot_token],
            )
        else:
            generations = oa_completion(
                engine=self.engine,
                prompt=context,
                max_tokens=max_length,
                temperature=0.,
                logprobs=5,
                stop=stopping_criteria,
            )
        return generations
