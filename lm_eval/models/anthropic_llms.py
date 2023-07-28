import os
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
import time
import anthropic
from lm_eval.logger import eval_logger
from typing import List, Literal


def anthropic_completion(
    client: anthropic.Anthropic,
    model: str,
    prompt: str,
    max_tokens_to_sample: int,
    temperature: float,
    stop: List[str],
):
    """Query Anthropic API for completion.

    Retry with back-off until they respond
    """
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
            return response.completion
        except anthropic.RateLimitError as e:
            eval_logger.warning(
                f"RateLimitError occurred: {e.__cause__}\n Retrying in {backoff_time} seconds"
            )
            time.sleep(backoff_time)
            backoff_time *= 1.5
        except anthropic.APIConnectionError as e:
            eval_logger.critical(f"Server unreachable: {e.__cause__}")
            break
        except anthropic.APIStatusError as e:
            eval_logger.critical(f"API error {e.status_code}: {e.message}")
            break


@register_model("anthropic")
class AnthropicLM(LM):
    REQ_CHUNK_SIZE = 20  # TODO: not used

    def __init__(
        self,
        batch_size=None,
        model: str = "claude-2.0",
        max_tokens_to_sample: int = 256,
        temperature: float = 0.0,
    ):  # TODO: remove batch_size
        """Anthropic API wrapper.

        :param model: str
            Anthropic model e.g. 'claude-instant-v1', 'claude-2'
        """
        super().__init__()

        self.model = model
        self.client = anthropic.Anthropic()
        self.temperature = temperature
        self.max_tokens_to_sample = max_tokens_to_sample
        self.tokenizer = self.client.get_tokenizer()

    @property
    def eot_token_id(self):
        # Not sure but anthropic.AI_PROMPT -> [203, 203, 50803, 30]
        raise NotImplementedError("No idea about anthropic tokenization.")

    @property
    def max_length(self):
        return 2048

    @property
    def max_gen_toks(self):
        return self.max_tokens_to_sample

    @property
    def batch_size(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError("No support for logits.")

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError("No support for logits.")

    def tok_encode(self, string: str) -> List[int]:
        return self.tokenizer.encode(string).ids

    def tok_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        raise NotImplementedError("No support for logits.")

    def greedy_until(self, requests):
        if not requests:
            return []

        requests = [req.args for req in requests]

        res = []
        for request in tqdm(requests):
            inp = request[0]
            request_args = request[1]
            until = request_args["until"]
            response = anthropic_completion(
                client=self.client,
                model=self.model,
                prompt=inp,
                max_tokens_to_sample=self.max_tokens_to_sample,
                temperature=self.temperature,  # TODO: implement non-greedy sampling for Anthropic
                stop=until,
            )
            res.append(response)

            self.cache_hook.add_partial("greedy_until", request, response)

        return res

    def _model_call(self, inps):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override greedy_until
        raise NotImplementedError()

    def loglikelihood(self, requests):
        raise NotImplementedError("No support for logits.")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("No support for logits.")
