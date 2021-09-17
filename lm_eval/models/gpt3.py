import os
import numpy as np
import transformers
from lm_eval.base import LM
from lm_eval import utils
from tqdm import tqdm
import time


def get_result(response, ctxlen):
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
    import openai

    backoff_time = 3
    while True:
        try:
            return openai.Completion.create(**kwargs)
        except openai.error.OpenAIError:
            time.sleep(backoff_time)
            backoff_time *= 1.5


class GPT3LM(LM):

    MAX_LENGTH = 2048
    REQ_CHUNK_SIZE = 20
    MAX_GEN_TOKS = 256

    def __init__(self, engine, truncate=False):
        """

        :param engine: str
            OpenAI API engine (e.g. davinci)
        :param truncate: bool
            Truncate input if too long (if False and input is too long, throw error)
        """
        super().__init__()
        import openai
        self.engine = engine
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')


        # to make the annoying "Using pad_token, but it is not set yet." error go away
        self.tokenizer.pad_token = "<|endoftext|>"
        assert self.tokenizer.encode('hello\n\nhello') == [31373, 198, 198, 31373]
        self.truncate = truncate
        self.end_of_text_token_id = self.tokenizer.convert_tokens_to_ids(["<|endoftext|>"])[0]

        # Read from environment variable OPENAI_API_SECRET_KEY
        openai.api_key = os.environ["OPENAI_API_SECRET_KEY"]

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config={}):
        args = utils.simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(**args, **args2)

    def loglikelihood(self, requests):
        new_reqs = []
        for context, continuation in requests:
            if context == "":
                # end of text as context
                context_enc = [50256]
            else:
                context_enc = self.tokenizer.encode(context)

            continuation_enc = self.tokenizer.encode(continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def loglikelihood_rolling(self, requests):
        # TODO: switch implementation to use _loglikelihood_tokens rather than having it do its own thing

        loglikelihoods = []
        for string, in tqdm(requests):
            encoded = self.tokenizer.encode_plus(string)["input_ids"]
            rolling_token_windows = utils.get_rolling_token_windows(
                token_list=encoded,
                prefix_token=self.end_of_text_token_id,
                max_seq_len=self.MAX_LENGTH,
                context_len=1,
            )
            string_loglikelihoods = []
            for input_tokens, pred_tokens in rolling_token_windows:
                block_output = self.get_token_logprobs(
                    input_tokens=input_tokens,
                    pred_tokens=pred_tokens,
                )
                string_loglikelihoods.append(block_output["logprobs"])
            string_loglikelihoods = np.concatenate(string_loglikelihoods).sum()
            loglikelihoods.append(string_loglikelihoods)

        return loglikelihoods

    def _loglikelihood_tokens(self, requests):
        import openai
        res = []

        def _collate(x):
            # this doesn't efficiently handle last-token differences yet, but those are kinda annoying because
            # it's not guaranteed that the 100 or so logprobs we get to see actually contain all the continuations
            # we care about and so we need some kind of backup for when it isn't
            toks = x[1] + x[2]
            return (-len(toks), tuple(toks))
        
        reord = utils.Reorderer(requests, _collate)

        for chunk in tqdm(list(utils.chunks(reord.get_reordered(), self.REQ_CHUNK_SIZE))):
            inps = []
            ctxlens = []
            for cache_key, context_enc, continuation_enc in chunk:
                inp = (context_enc + continuation_enc)[-self.MAX_LENGTH:]
                ctxlen = len(context_enc) - max(0, len(context_enc) + len(continuation_enc) - self.MAX_LENGTH)

                inps.append(inp)
                ctxlens.append(ctxlen)

            response = oa_completion(
                engine=self.engine,
                prompt=inps,
                echo=True,
                max_tokens=0, temperature=0.,
                logprobs=10,
            )

            for resp, ctxlen, (cache_key, context_enc, continuation_enc) in zip(response.choices, ctxlens, chunk):
                answer = get_result(resp, ctxlen)

                res.append(answer)

                # partial caching
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)

        return reord.get_original(res)

    def get_token_logprobs(self, input_tokens, pred_tokens):
        pred_start = len(input_tokens) - len(pred_tokens) + 1
        # We're going to stitch together the input_tokens and pred_tokens
        # In the longest case, this gets us to length = max_seq_len+1 (which the API works with)
        assert input_tokens[pred_start:] == pred_tokens[:-1]
        token_ids = input_tokens + [pred_tokens[-1]]
        response = oa_completion(
            engine=self.engine,
            prompt=token_ids,
            max_tokens=0,
            temperature=0.0,
            logprobs=0,
            echo=True,
        )
        logprobs = np.array(response["choices"][0]["logprobs"]["token_logprobs"][pred_start:])
        positions = np.arange(pred_start-1, pred_start-1 + len(token_ids[pred_start:]))
        return {
            "logprobs": logprobs,
            "positions": positions,
        }

    def greedy_until(self, requests):
        if not requests: return []
        import openai
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return (len(toks), x[0])
        
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
            
            if ret: yield ret, lastuntil

        # todo: more intelligent batching for heterogenous `until`
        for chunk, until in tqdm(list(sameuntil_chunks(reord.get_reordered(), self.REQ_CHUNK_SIZE))):
            inps = []
            for context, _ in chunk:
                context_enc = self.tokenizer.encode(context)
                inp = context_enc[-(self.MAX_LENGTH - self.MAX_GEN_TOKS):]
                inps.append(inp)

            response = oa_completion(
                engine=self.engine,
                prompt=inps,
                max_tokens=self.MAX_GEN_TOKS, 
                temperature=0.,
                logprobs=10,
                stop=until
            )

            for resp, (context, until) in zip(response.choices, chunk):
                s = resp['text']

                for term in until:
                    s = s.split(term)[0]

                # partial caching
                self.cache_hook.add_partial("greedy_until", (context, until), s)
                
                res.append(s)
        
        return reord.get_original(res)
