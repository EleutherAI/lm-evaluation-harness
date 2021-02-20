import os
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
        import openai
        self.engine = engine
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')


        # to make the annoying "Using pad_token, but it is not set yet." error go away
        self.tokenizer.pad_token = "<|endoftext|>"
        assert self.tokenizer.encode('hello\n\nhello') == [31373, 198, 198, 31373]
        self.truncate = truncate

        # Read from environment variable OPENAI_API_SECRET_KEY
        openai.api_key = os.environ["OPENAI_API_SECRET_KEY"]

    @classmethod
    def create_from_arg_string(cls, arg_string):
        args = utils.simple_parse_args_string(arg_string)
        return cls(engine=args.get("engine", "davinci"))

    def loglikelihood(self, requests):
        import openai
        res = []

        for chunk in tqdm(list(utils.chunks(requests, self.REQ_CHUNK_SIZE))):
            inps = []
            ctxlens = []
            for context, continuation in chunk:
                if context == "":
                    # end of text as context
                    context_enc = [50256]
                else:
                    context_enc = self.tokenizer.encode(context)
                    
                continuation_enc = self.tokenizer.encode(continuation)
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

            for resp, ctxlen in zip(response.choices, ctxlens):
                res.append(get_result(resp, ctxlen))
            
        return res

    def greedy_until(self, requests):
        if not requests: return []
        import openai
        res = []

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
        for chunk, until in tqdm(list(sameuntil_chunks(requests, self.REQ_CHUNK_SIZE))):
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

            for resp in response.choices:
                s = response.choices[0]['text']

                for term in until:
                    s = s.split(term)[0]

                res.append(s)
        
        return res

