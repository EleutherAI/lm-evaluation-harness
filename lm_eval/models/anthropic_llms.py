import logging
import os
from functools import cached_property
from typing import Any, Dict, List, Tuple, Union

from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.openai_completions import LocalCompletionsAPI
from lm_eval.models.utils import handle_stop_sequences, retry_on_specific_exceptions


eval_logger = logging.getLogger(__name__)


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
    except ModuleNotFoundError as exception:
        raise type(exception)(
            "attempted to use 'anthropic' LM type, but package `anthropic` is not installed. \
please install anthropic via `pip install 'lm-eval[anthropic]'` or `pip install -e '.[anthropic]'`",
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


def anthropic_chat(
    client,  #: anthropic.Anthropic,
    model: str,
    prompt: str,
    max_tokens: int,
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
            Anthropic model e.g. 'claude-3-opus-20240229', 'claude-3-sonnet-20240229'
        prompt: str
            Prompt to feed to the model
        max_tokens: int
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
    except ModuleNotFoundError as exception:
        raise type(exception)(
            "attempted to use 'anthropic' LM type, but package `anthropic` is not installed. \
please install anthropic via `pip install 'lm-eval[anthropic]'` or `pip install -e '.[anthropic]'`",
        )

    def _exception_callback(e: Exception, sleep_time: float) -> None:
        eval_logger.warning(
            f"RateLimitError occurred: {e.__cause__}\n Retrying in {sleep_time} seconds"
        )

    @retry_on_specific_exceptions(
        on_exceptions=[
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
            anthropic.APIStatusError,
        ],
        max_retries=None,  # retry forever, consider changing
        on_exception_callback=_exception_callback,
    )
    def messages():
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": f"{prompt}"}],
            **kwargs,
        )
        return response.content[0].text

    return messages()


@register_model("anthropic-completions")
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
        except ModuleNotFoundError as exception:
            raise type(exception)(
                "attempted to use 'anthropic' LM type, but package `anthropic` is not installed. \
please install anthropic via `pip install 'lm-eval[anthropic]'` or `pip install -e '.[anthropic]'`",
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

    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        try:
            import anthropic
        except ModuleNotFoundError as exception:
            raise type(exception)(
                "attempted to use 'anthropic' LM type, but package `anthropic` is not installed. \
please install anthropic via `pip install 'lm-eval[anthropic]'` or `pip install -e '.[anthropic]'`",
            )

        if not requests:
            return []

        _requests: List[Tuple[str, dict]] = [req.args for req in requests]

        res = []
        for request in tqdm(_requests, disable=disable_tqdm):
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

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("No support for logits.")

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("No support for logits.")


@register_model("anthropic-chat", "anthropic-chat-completions")
class AnthropicChat(LocalCompletionsAPI):
    def __init__(
        self,
        base_url="https://api.anthropic.com/v1/messages",
        tokenizer_backend=None,
        **kwargs,
    ):
        super().__init__(
            base_url=base_url, tokenizer_backend=tokenizer_backend, **kwargs
        )
        eval_logger.warning(
            "Chat completions does not support batching. Defaulting to batch size 1."
        )
        self._batch_size = 1
        self.anthropic_version = "2023-06-01"
        eval_logger.warning(
            f"Using Anthropic Version: {self.anthropic_version}. Confirm the current version here: https://docs.anthropic.com/en/api/versioning"
        )

    @cached_property
    def api_key(self):
        """Override this property to return the API key for the API request."""
        key = os.environ.get("ANTHROPIC_API_KEY", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the ANTHROPIC_API_KEY environment variable."
            )
        return key

    @cached_property
    def header(self):
        return {
            "x-api-key": f"{self.api_key}",
            "anthropic-version": self.anthropic_version,
        }

    def _create_payload(
        self,
        messages: List[Dict],
        generate=True,
        gen_kwargs: dict = None,
        eos="\n\nHuman:",
        **kwargs,
    ) -> dict:
        system = (
            messages[0].get("content") if messages[0].get("role") == "system" else None
        )
        if system:
            messages = messages[1:]

        cleaned_messages = []
        for msg in messages:
            cleaned_msg = {
                "role": msg["role"],
                "content": [
                    {"type": msg["type"], msg["type"]: msg["content"]},
                ],
            }
            cleaned_messages.append(cleaned_msg)

        gen_kwargs.pop("do_sample", False)
        max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
        temperature = gen_kwargs.pop("temperature", 0)
        stop = handle_stop_sequences(gen_kwargs.pop("until", ["\n\nHuman:"]), eos=eos)
        if not isinstance(stop, list):
            stop = [stop]

        # Filter out empty or whitespace-only stop sequences for Anthropic API
        stop = [s for s in stop if s and s.strip()]

        out = {
            "messages": cleaned_messages,
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop_sequences": stop,
            **gen_kwargs,
        }
        if system:
            out["system"] = system
        return out

    def parse_generations(
        self, outputs: Union[Dict, List[Dict]], **kwargs
    ) -> List[str]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            for choices in out["content"]:
                res.append(choices["text"])
        return res

    def tok_encode(
        self,
        string: str,
        left_truncate_len=None,
        add_special_tokens=None,
        **kwargs,
    ) -> List[str]:
        return [string]

    def loglikelihood(self, requests, **kwargs):
        raise NotImplementedError(
            "Anthropic Chat Completions API does not support the return of loglikelihood"
        )
