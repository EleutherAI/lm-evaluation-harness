import os
import transformers
from lm_eval.base import LM
from lm_eval import utils


class GPT3LM(LM):

    MAX_LENGTH = 2048

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
        self.truncate = truncate

        # Read from environment variable OPENAI_API_SECRET_KEY
        openai.api_key = os.environ["OPENAI_API_SECRET_KEY"]

    @classmethod
    def create_from_arg_string(cls, arg_string):
        args = utils.simple_parse_args_string(arg_string)
        return cls(engine=args.get("engine", "davinci"))

    def loglikelihood(self, context, continuation):
        import openai
        
        context_enc = self.tokenizer.encode(context)
        continuation_enc = self.tokenizer.encode(continuation)
        inp = (context_enc + continuation_enc)[-1024:]
        ctxlen = len(context_enc) - max(0, len(context_enc) + len(continuation_enc) - 1024)

        response = openai.Completion.create(
            engine=self.engine,
            prompt=inp,
            echo=True,
            max_tokens=0, temperature=0.0,
            logprobs=0,
        )
        logprobs = response.choices[0]["logprobs"]["token_logprobs"]
        continuation_logprobs = logprobs[ctxlen:]
        return sum(continuation_logprobs)
