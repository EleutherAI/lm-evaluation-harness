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
        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
        self.truncate = truncate

        # Read from environment variable OPENAI_API_SECRET_KEY
        openai.api_key = os.environ["OPENAI_API_SECRET_KEY"]

    @classmethod
    def create_from_arg_string(cls, arg_string):
        args = utils.simple_parse_args_string(arg_string)
        return cls(engine=args.get("engine", "davinci"))

    def generate(self, context, max_gen_length):
        import openai
        if self.truncate:
            prompt = self.smart_truncate(context, buffer=max_gen_length)
        else:
            prompt = context

        response = openai.Completion.create(
            engine=self.engine,
            prompt=prompt,
            max_tokens=max_gen_length,
            temperature=0.0,
        )
        return response.choices[0]["text"]

    def loglikelihood(self, context, continuation):
        import openai
        full_text = context + continuation
        full_text_length = len(self.tokenizer.tokenize(full_text))
        context_length = len(self.tokenizer.tokenize(context))
        continuation_length = len(self.tokenizer.tokenize(continuation))
        assert full_text_length == context_length + continuation_length
        if self.truncate:
            prompt = self.smart_truncate(full_text, buffer=0)
        else:
            prompt = full_text
        response = openai.Completion.create(
            engine=self.engine,
            prompt=prompt,
            echo=True,
            max_tokens=0, temperature=0.0,
            logprobs=0,
        )
        logprobs = response.choices[0]["logprobs"]["token_logprobs"]
        continuation_logprobs = logprobs[-continuation_length:]
        return sum(continuation_logprobs)

    def smart_truncate(self, string, buffer=1):
        tokens = self.tokenizer.tokenize(string)
        available_length = self.MAX_LENGTH - 1 - buffer  # OpenAI adds 1 token
        kept_tokens = tokens[-available_length:]
        new_string = self.tokenizer.convert_tokens_to_string(kept_tokens)
        return new_string

    def num_tokens(self, string):
        return len(self.tokenizer.tokenize(string))
