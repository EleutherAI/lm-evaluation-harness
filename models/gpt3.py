import os
import openai
import transformers
from ..base import LM
from .. import utils
from . import MODEL_REGISTRY


@MODEL_REGISTRY.register("gpt3")
class GPT3LM(LM):
    def __init__(self, engine):
        self.engine = engine
        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
        # Read from environment variable OPENAI_API_SECRET_KEY
        openai.api_key = os.environ["OPENAI_API_SECRET_KEY"]

    @classmethod
    def create_from_args(cls, arg_string):
        args = utils.simple_parse_args_string(arg_string)
        return cls(engine=args.get("engine", "davinci"))

    def generate(self, context, max_gen_length):
        response = openai.Completion.create(
            engine=self.engine,
            prompt=context,
            max_tokens=max_gen_length,
            temperature=0.0,
        )
        return response.choices[0]["text"]

    def loglikelihood(self, context, continuation):
        full_text = context + continuation
        full_text_length = len(self.tokenizer.tokenize(full_text))
        context_length = len(self.tokenizer.tokenize(context))
        continuation_length = len(self.tokenizer.tokenize(continuation))
        assert full_text_length == context_length + continuation_length
        response = openai.Completion.create(
            engine=self.engine,
            prompt=full_text,
            max_tokens=0, temperature=0.0,
            logprobs=0,
        )
        logprobs = response.choices[0]["logprobs"]["token_logprobs"]
        continuation_logprobs = logprobs[-continuation_length:]
        return sum(continuation_logprobs)
