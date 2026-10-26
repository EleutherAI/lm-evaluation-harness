import json
import logging
import os
from functools import cached_property
from operator import itemgetter
from typing import Any

from lm_eval.api.registry import register_model
from lm_eval.models.api_models import JsonChatStr, TemplateAPI
from lm_eval.models.utils import handle_stop_sequences, postprocess_generated_text


eval_logger = logging.getLogger(__name__)


class CompletionText(str):
    """Generated text with API completion metadata for sample logging."""

    def __new__(
        cls,
        text: str,
        answer_found: bool,
        stop_reason: str | None = None,
        full_resp: str | None = None,
    ):
        result = super().__new__(cls, text)
        result.answer_found = answer_found
        result.stop_reason = stop_reason
        result.full_resp = full_resp if full_resp is not None else text
        return result

    def __reduce__(self):
        return (
            self.__class__,
            (str(self), self.answer_found, self.stop_reason, self.full_resp),
        )


@register_model("local-completions")
class LocalCompletionsAPI(TemplateAPI):
    def __init__(
        self,
        base_url=None,
        tokenizer_backend="auto",
        verify_certificate=True,
        ca_cert_path=None,
        auth_token=None,
        think_end_token: str | None = None,
        **kwargs,
    ):
        # Auto-detect tokenizer backend
        if tokenizer_backend == "auto":
            if base_url:
                from lm_eval.utils import check_remote_tokenizer_support

                if check_remote_tokenizer_support(
                    base_url,
                    verify_certificate=verify_certificate,
                    ca_cert_path=ca_cert_path,
                    auth_token=auth_token,
                ):
                    eval_logger.info(
                        "Auto-detected remote tokenizer support. Using remote tokenizer backend."
                    )
                    tokenizer_backend = "remote"
                else:
                    eval_logger.info(
                        "Remote tokenizer not supported. Using huggingface tokenizer backend."
                    )
                    tokenizer_backend = "huggingface"
            else:
                eval_logger.warning(
                    "No base_url provided. Using huggingface tokenizer backend."
                )
                tokenizer_backend = "huggingface"

        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            verify_certificate=verify_certificate,
            ca_cert_path=ca_cert_path,
            auth_token=auth_token,
            **kwargs,
        )
        self.think_end_token = think_end_token

    def _create_payload(
        self,
        messages: list[list[int]] | list[dict] | list[str] | str,
        generate=False,
        gen_kwargs: dict | None = None,
        seed: int = 1234,
        eos=None,
        **kwargs,
    ) -> dict:
        if generate:
            gen_kwargs.pop("do_sample", False)
            if "max_tokens" in gen_kwargs:
                max_tokens = gen_kwargs.pop("max_tokens")
            else:
                max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
            temperature = gen_kwargs.pop("temperature", 0)
            task_until = gen_kwargs.pop("until", None)
            stop = handle_stop_sequences(task_until, eos)
            if self.think_end_token:
                # Task stops may occur in the reasoning trace. Apply them to
                # the answer after removing the trace instead.
                stop = [sequence for sequence in stop if sequence == eos]
            return {
                "prompt": messages,
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": stop,
                "seed": seed,
                **gen_kwargs,
            }
        else:
            return {
                "model": self.model,
                "prompt": messages,
                "temperature": 0,
                "max_tokens": 1,
                "logprobs": 1,
                "seed": seed,
                "echo": True,
            }

    @staticmethod
    def parse_logprobs(
        outputs: dict | list[dict],
        tokens: list[list[int]] | None = None,
        ctxlens: list[int] | None = None,
        **kwargs,
    ) -> list[tuple[float, bool]]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            for choice, ctxlen in zip(
                sorted(out["choices"], key=itemgetter("index")),
                ctxlens,
                strict=False,
            ):
                assert ctxlen > 0, "Context length must be greater than 0"
                logprobs = sum(choice["logprobs"]["token_logprobs"][ctxlen:-1])
                tokens_logprobs = choice["logprobs"]["token_logprobs"][ctxlen:-1]
                top_logprobs = choice["logprobs"]["top_logprobs"][ctxlen:-1]
                is_greedy = True
                for tok, top in zip(tokens_logprobs, top_logprobs, strict=False):
                    if tok != max(top.values()):
                        is_greedy = False
                        break
                res.append((logprobs, is_greedy))
        return res

    def _to_completion_text(
        self,
        text: str,
        choice: dict,
        stop: str | list[str] | None = None,
    ) -> CompletionText:
        """Attach the metadata that ends up in `samples_*.jsonl`.

        Shared by the completions and chat clients so both log the same
        `stop_reasons` / `found_answers` / `full_resps` fields. `text` is the raw
        generation, thinking content included; the returned string is the answer
        after stop-sequence and thinking-trace post-processing, while `full_resp`
        keeps the untrimmed generation.
        """
        finish_reason = choice.get("finish_reason")
        api_stop_reason = choice.get("stop_reason")
        stop_sequences = [stop] if isinstance(stop, str) else (stop or [])
        matched_stop = next(
            (
                sequence
                for sequence in stop_sequences
                if sequence and text.endswith(sequence)
            ),
            None,
        )
        inferred_eos = (
            finish_reason == "stop"
            and api_stop_reason is None
            and matched_stop is None
            and self.eos_string is not None
        )
        if finish_reason == "length":
            stop_reason = "max_gen_toks"
        elif isinstance(api_stop_reason, str):
            stop_reason = f"stop_sequence:{api_stop_reason}"
        elif isinstance(api_stop_reason, int):
            stop_text = (
                self.tokenizer.decode([api_stop_reason])
                if self.tokenizer is not None
                else str(api_stop_reason)
            )
            stop_reason = f"stop_sequence:{stop_text}"
        elif matched_stop is not None:
            stop_reason = f"stop_sequence:{matched_stop}"
        elif inferred_eos:
            stop_reason = f"stop_sequence:{self.eos_string}"
        elif finish_reason == "stop":
            stop_reason = "natural"
        else:
            stop_reason = finish_reason
        answer = postprocess_generated_text(
            text,
            stop=stop if self.think_end_token else None,
            think_end_token=self.think_end_token,
        )
        return CompletionText(
            answer,
            not self.think_end_token or self.think_end_token in text,
            stop_reason,
            text,
        )

    def parse_generations(
        self,
        outputs: dict | list[dict],
        stop: str | list[str] | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> list[str]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            completion_tokens = (out.get("usage") or {}).get("completion_tokens")
            if max_tokens is not None and completion_tokens is not None:
                # OpenAI-compatible completion usage counts all choices in the
                # response; each choice has its own max_tokens limit.
                limit = max_tokens * len(out["choices"])
                assert len(out["choices"]) == 1
                assert completion_tokens <= limit, (
                    f"Completion API generated {completion_tokens} tokens across "
                    f"{len(out['choices'])} choices, exceeding the requested "
                    f"max_tokens={max_tokens} per choice."
                )
            tmp = [None] * len(out["choices"])
            for choice in out["choices"]:
                text = choice["text"]
                if text is None:
                    continue
                completion_text = self._to_completion_text(text, choice, stop)
                tmp[choice["index"]] = completion_text
            res = res + tmp
        return res

    @property
    def api_key(self):
        return os.environ.get("OPENAI_API_KEY", "")


@register_model("local-chat-completions")
class LocalChatCompletion(LocalCompletionsAPI):
    """
    Minimal chat-completions wrapper.
    - Only accepts messages as list[dict].
    - No tokenization or template logic.
    - Use with --apply_chat_template or ensure upstream formats messages correctly.
    """

    def __init__(
        self,
        base_url=None,
        tokenizer_backend=None,
        tokenized_requests=None,
        think_end_token: str | None = None,
        verify_certificate=True,
        ca_cert_path=None,
        auth_token=None,
        **kwargs,
    ):
        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            verify_certificate=verify_certificate,
            ca_cert_path=ca_cert_path,
            auth_token=auth_token,
            **kwargs,
        )
        self.think_end_token = think_end_token
        if self._batch_size > 1:
            eval_logger.warning(
                "Chat completions does not support batching. Defaulting to batch size 1."
            )
            self._batch_size = 1

    def apply_chat_template(
        self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True
    ) -> JsonChatStr:
        # Chat completions accepts messages directly, including when a tokenizer
        # is loaded for other purposes.
        return JsonChatStr(json.dumps(chat_history, ensure_ascii=False))

    def _create_payload(
        self,
        messages: list[dict],
        generate=False,
        gen_kwargs: dict | None = None,
        seed=1234,
        eos=None,
        **kwargs,
    ) -> dict:
        assert isinstance(messages, list) and all(
            isinstance(m, dict) for m in messages
        ), (
            "LocalChatCompletion expects messages as list[dict]. "
            "If you see this error, ensure --apply_chat_template is set or upstream code formats messages correctly."
        )
        gen_kwargs = gen_kwargs or {}
        gen_kwargs.pop("do_sample", False)
        if "max_tokens" in gen_kwargs:
            max_tokens = gen_kwargs.pop("max_tokens")
        else:
            max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
        temperature = gen_kwargs.pop("temperature", 0)
        stop = handle_stop_sequences(gen_kwargs.pop("until", None), eos)
        if not isinstance(stop, (list, tuple)):
            stop = [stop]
        return {
            "messages": messages,
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop[:4],
            "seed": seed,
            **gen_kwargs,
        }

    def parse_generations(
        self,
        outputs: dict | list[dict],
        stop: str | list[str] | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> list[str]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            try:
                tmp = [None] * len(out["choices"])
                for choice in out["choices"]:
                    content = choice["message"]["content"]
                    if content is None:
                        continue
                    # Same metadata as the completions client, so the sample
                    # logs carry identical fields; `full_resp` keeps the raw
                    # generation, thinking trace included.
                    tmp[choice["index"]] = self._to_completion_text(
                        content, choice, stop
                    )
            except (IndexError, KeyError, TypeError) as e:
                # account for cases that generation is blocked by content filter,
                # which is common for Azure OpenAI Service,
                # not sure if need to account for multiple choices
                eval_logger.warning(f"Could not parse generations: {e}")
                tmp = [""]
            res = res + tmp
        return res

    def tok_encode(
        self,
        string: str | Any,
        left_truncate_len=None,
        add_special_tokens=None,
        **kwargs,
    ) -> list[str] | list[int] | Any:
        return string

    def loglikelihood(self, requests, **kwargs):
        raise NotImplementedError(
            "Loglikelihood is not supported for chat completions. Consider using the completions API instead."
        )


@register_model(
    "openai-completions",
)
class OpenAICompletionsAPI(LocalCompletionsAPI):
    def __init__(
        self,
        base_url="https://api.openai.com/v1/completions",
        tokenizer_backend="tiktoken",
        **kwargs,
    ):
        super().__init__(
            base_url=base_url, tokenizer_backend=tokenizer_backend, **kwargs
        )

    @cached_property
    def api_key(self):
        """Override this property to return the API key for the API request."""
        key = os.environ.get("OPENAI_API_KEY", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the `OPENAI_API_KEY` environment variable."
            )
        return key

    def loglikelihood(self, requests, **kwargs):
        assert self.model in [
            "babbage-002",
            "davinci-002",
        ], (
            f"Prompt loglikelihoods are only supported by OpenAI's API for {['babbage-002', 'davinci-002']}."
        )
        return super().loglikelihood(requests, **kwargs)

    def chat_template(self, chat_template: bool | str = False) -> str | None:
        return ""


@register_model("openai-chat-completions")
class OpenAIChatCompletion(LocalChatCompletion):
    def __init__(
        self,
        base_url="https://api.openai.com/v1/chat/completions",
        tokenizer_backend=None,
        tokenized_requests=False,
        **kwargs,
    ):
        if "o1" in kwargs.get("model", ""):
            eval_logger.warning(
                "o1 models do not support `stop` and only support temperature=1"
            )

        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            **kwargs,
        )

    @cached_property
    def api_key(self):
        """Override this property to return the API key for the API request."""
        key = os.environ.get("OPENAI_API_KEY", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the `OPENAI_API_KEY` environment variable."
            )
        return key

    def loglikelihood(self, requests, **kwargs):
        raise NotImplementedError(
            "Loglikelihood (and therefore `multiple_choice`-type tasks) is not supported for chat completions as OpenAI does not provide prompt logprobs. See https://github.com/EleutherAI/lm-evaluation-harness/issues/942#issuecomment-1777836312 or https://github.com/EleutherAI/lm-evaluation-harness/issues/1196 for more background on this limitation."
        )

    def _create_payload(
        self,
        messages: list[dict],
        generate=False,
        gen_kwargs: dict | None = None,
        seed=1234,
        eos="<|endoftext|>",
        **kwargs,
    ) -> dict:
        assert type(messages) is not str, (
            "chat-completions require the --apply_chat_template flag."
        )
        gen_kwargs.pop("do_sample", False)
        if "max_tokens" in gen_kwargs:
            max_tokens = gen_kwargs.pop("max_tokens")
        else:
            max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
        temperature = gen_kwargs.pop("temperature", 0)
        stop = handle_stop_sequences(gen_kwargs.pop("until", ["<|endoftext|>"]), eos)
        if not isinstance(stop, (list, tuple)):
            stop = [stop]
        output = {
            "messages": messages,
            "model": self.model,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop[:4],
            "seed": seed,
            **gen_kwargs,
        }
        if (
            "o1" in self.model
            or "5" in self.model
            or "o3" in self.model
            or "o4" in self.model
        ):
            output.pop("stop")
            output["temperature"] = 1
        return output


@register_model("azure-openai-chat-completions")
class AzureOpenaiChatCompletionsLM(OpenAIChatCompletion):
    def __init__(
        self,
        model: str = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        base_url: str = os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview"),
        truncate: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        try:
            import openai
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "attempted to use 'openai' LM type, but package `openai` or `tiktoken` are not installed. \
    please install these via `pip install lm-eval[openai]` or `pip install -e .[openai]`",
            ) from exc
        self.model = model
        self.base_url = f"{base_url}/openai/deployments/{model}/chat/completions?api-version={api_version}"
        self.truncate = truncate
        self.client = openai.AzureOpenAI(
            azure_endpoint=base_url, api_version=api_version, api_key=self.api_key
        )

    @cached_property
    def api_key(self):
        key = os.environ.get("AZURE_OPENAI_API_KEY", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the `AZURE_OPENAI_API_KEY` environment variable."
            )
        return key
