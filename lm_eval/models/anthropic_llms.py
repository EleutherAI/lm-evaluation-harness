import os
from lm_eval.base import BaseLM
from tqdm import tqdm
import time


def anthropic_completion(
    client, model, prompt, max_tokens_to_sample, temperature, stop
):
    """Query Anthropic API for completion.

    Retry with back-off until they respond
    """
    import anthropic

    backoff_time = 3
    while True:
        try:
            response = client.completions.create(
                prompt=f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}",
                model=model,
                # NOTE: Claude really likes to do CoT, and overly aggressive stop sequences
                #       (e.g. gsm8k's ":") may truncate a lot of the input.
                stop_sequences=[anthropic.HUMAN_PROMPT] + stop,
                max_tokens_to_sample=max_tokens_to_sample,
                temperature=temperature,
            )
            print(response)
            return response.completion
        except RuntimeError:
            # TODO: I don't actually know what error Anthropic raises when it times out
            #       So err update this error when we find out.
            import traceback

            traceback.print_exc()
            time.sleep(backoff_time)
            backoff_time *= 1.5


class AnthropicLM(BaseLM):
    REQ_CHUNK_SIZE = 20

    def __init__(self, model="claude-2"):
        """

        :param model: str
            Anthropic model e.g. claude-instant-v1
        """
        super().__init__()
        import anthropic

        self.model = model
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    @property
    def eot_token_id(self):
        raise NotImplementedError("No idea about anthropic tokenization.")

    @property
    def max_length(self):
        return 2048

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def tok_encode(self, string: str):
        raise NotImplementedError("No idea about anthropic tokenization.")

    def tok_decode(self, tokens):
        raise NotImplementedError("No idea about anthropic tokenization.")

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        raise NotImplementedError("No support for logits.")

    def greedy_until(self, requests):
        if not requests:
            return []

        res = []
        for request in tqdm(requests):
            inp = request[0]
            request_args = request[1]
            until = request_args["until"]
            response = anthropic_completion(
                client=self.client,
                model=self.model,
                prompt=inp,
                max_tokens_to_sample=self.max_gen_toks,
                temperature=0.0,
                stop=until,
            )
            res.append(response)
        return res

    def _model_call(self, inps):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override greedy_until
        raise NotImplementedError()
