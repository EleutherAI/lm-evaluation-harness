from typing import Any, List, Tuple

from tqdm import tqdm

from lm_eval import utils
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import retry_on_specific_exceptions


eval_logger = utils.eval_logger


def anthropic_completion(
    client,  #: anthropic.Anthropic,
    model: str,
    prompt: str,
    max_tokens_to_sample: int,
    temperature: float,
    stop: List[str],
    **kwargs: Any,
) -> str:
    """Wrapper function around the Anthropic completion API client with exponential back-off
    in case of RateLimitError.

    params:
        client: anthropic.Anthropic
            Anthropic API client
        model: str
            Anthropic model e.g. 'claude-instant-v1', 'claude-2'
        prompt: str
            Prompt to feed to the model
        max_tokens_to_sample: int
            Maximum number of tokens to sample from the model
        temperature: float
            Sampling temperature
        stop: List[str]
            List of stop sequences
        kwargs: Any
            Additional model_args to pass to the API client
    """

    try:
        import anthropic
    except ModuleNotFoundError:
        raise Exception(
            "attempted to use 'anthropic' LM type, but package `anthropic` is not installed. \
please install anthropic via `pip install lm-eval[anthropic]` or `pip install -e .[anthropic]`",
        )

    def _exception_callback(e: Exception, sleep_time: float) -> None:
        eval_logger.warning(
            f"RateLimitError occurred: {e.__cause__}\n Retrying in {sleep_time} seconds"
        )

    @retry_on_specific_exceptions(
        on_exceptions=[anthropic.RateLimitError],
        max_retries=None,  # retry forever, consider changing
        on_exception_callback=_exception_callback,
    )
    def completion():
        response = client.completions.create(
            prompt=f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}",
            model=model,
            # NOTE: Claude really likes to do CoT, and overly aggressive stop sequences
            #       (e.g. gsm8k's ":") may truncate a lot of the input.
            stop_sequences=[anthropic.HUMAN_PROMPT] + stop,
            max_tokens_to_sample=max_tokens_to_sample,
            temperature=temperature,
            **kwargs,
        )
        return response.completion

    return completion()


@register_model("anthropic")
class AnthropicLM(LM):
    REQ_CHUNK_SIZE = 20  # TODO: not used

    def __init__(
        self,
        batch_size: int = 1,
        model: str = "claude-2.0",
        max_tokens_to_sample: int = 256,
        temperature: float = 0,  # defaults to 1
        **kwargs,  # top_p, top_k, etc.
    ) -> None:
        """Anthropic API wrapper.

        :param model: str
            Anthropic model e.g. 'claude-instant-v1', 'claude-2'
        :param max_tokens_to_sample: int
            Maximum number of tokens to sample from the model
        :param temperature: float
            Sampling temperature
        :param kwargs: Any
            Additional model_args to pass to the API client
        """
        super().__init__()

        try:
            import anthropic
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'anthropic' LM type, but package `anthropic` is not installed. \
please install anthropic via `pip install lm-eval[anthropic]` or `pip install -e .[anthropic]`",
            )

        self.model = model
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic()
        self.temperature = temperature
        self.max_tokens_to_sample = max_tokens_to_sample
        self.tokenizer = self.client.get_tokenizer()
        self.kwargs = kwargs

    @property
    def eot_token_id(self):
        # Not sure but anthropic.HUMAN_PROMPT ?
        raise NotImplementedError("No idea about anthropic tokenization.")

    @property
    def max_length(self) -> int:
        return 2048

    @property
    def max_gen_toks(self) -> int:
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

    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("No support for logits.")

    def generate_until(self, requests) -> List[str]:
        try:
            import anthropic
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'anthropic' LM type, but package `anthropic` is not installed. \
please install anthropic via `pip install lm-eval[anthropic]` or `pip install -e .[anthropic]`",
            )

        if not requests:
            return []

        _requests: List[Tuple[str, dict]] = [req.args for req in requests]

        res = []
        for request in tqdm(_requests):
            try:
                inp = request[0]
                request_args = request[1]
                # generation_kwargs
                until = request_args.get("until")
                max_gen_toks = request_args.get("max_gen_toks", self.max_length)
                temperature = request_args.get("temperature", self.temperature)
                response = anthropic_completion(
                    client=self.client,
                    model=self.model,
                    prompt=inp,
                    max_tokens_to_sample=max_gen_toks,
                    temperature=temperature,  # TODO: implement non-greedy sampling for Anthropic
                    stop=until,  # type: ignore
                    **self.kwargs,
                )
                res.append(response)

                self.cache_hook.add_partial("generate_until", request, response)
            except anthropic.APIConnectionError as e:  # type: ignore # noqa: F821
                eval_logger.critical(f"Server unreachable: {e.__cause__}")
                break
            except anthropic.APIStatusError as e:  # type: ignore # noqa: F821
                eval_logger.critical(f"API error {e.status_code}: {e.message}")
                break

        return res

    def _model_call(self, inps):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override generate_until
        raise NotImplementedError()

    def loglikelihood(self, requests):
        raise NotImplementedError("No support for logits.")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("No support for logits.")
