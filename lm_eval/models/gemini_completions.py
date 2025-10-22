import copy
import logging
import os
from functools import cached_property
from operator import itemgetter
from typing import Any, Dict, List, Optional, Tuple, Union

from google import genai

from lm_eval.api.registry import register_model
from lm_eval.models.api_models import TemplateAPI, JsonChatStr
from lm_eval.models.utils import handle_stop_sequences


eval_logger = logging.getLogger(__name__)


@register_model("gemini-completions")
class GeminiCompletionsAPI(TemplateAPI):
    def __init__(
        self,
        base_url=None,
        tokenizer_backend: str = None, # Set to None as Gemini handles its own tokenization
        **kwargs,
    ):
        super().__init__(
            base_url=None, tokenizer_backend=tokenizer_backend, **kwargs
        )
        self.client = genai.Client(api_key=self.api_key)

    @cached_property
    def api_key(self):
        """Override this property to return the API key for the API request."""
        key = os.environ.get("OPENAI_API_KEY", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the `OPENAI_API_KEY` environment variable."
            )
        return key

    def tok_encode(
        self,
        string: Union[str, Any],
        left_truncate_len=None,
        add_special_tokens=None,
        **kwargs,
    ) -> Union[List[str], List[int], Any]:
        # Gemini API handles its own tokenization, so we can return the string directly
        return string

    def tok_decode(
        self,
        tokens: Union[List[int], Any],
        **kwargs,
    ) -> str:
        # Gemini API handles its own tokenization, so we can return the tokens directly
        return tokens

    def _create_payload(
        self,
        messages: Union[List[List[int]], List[dict], List[str], str],
        generate=False,
        gen_kwargs: Optional[dict] = None,
        seed: int = 1234,
        eos=None,
        **kwargs,
    ) -> dict:
        if generate:
            gen_kwargs.pop("do_sample", False)
            if "max_tokens" in gen_kwargs:
                max_output_tokens = gen_kwargs.pop("max_tokens")
            else:
                max_output_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
            temperature = gen_kwargs.pop("temperature", 0)
            stop_sequences = handle_stop_sequences(gen_kwargs.pop("until", None), eos)

            generation_config = {
                "max_output_tokens": max_output_tokens,
                "temperature": temperature,
                "stop_sequences": stop_sequences,
                "candidate_count": 1, # Gemini always returns 1 candidate for now
                **gen_kwargs,
            }
            return {
                "contents": format_gemini_contents(messages),
                "generation_config": generation_config,
            }
        else:
            raise NotImplementedError(
                "Loglikelihood is not supported for Gemini completions."
            )

    def model_call(
        self,
        messages: Union[List[List[int]], List[str], List[JsonChatStr]],
        *,
        generate: bool = True,
        gen_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> Optional[dict]:
        gen_kwargs = copy.deepcopy(gen_kwargs)
        payload = self._create_payload(
            self.create_message(messages),
            generate=generate,
            gen_kwargs=gen_kwargs,
            seed=self._seed,
            eos=self.eos_string,
            **kwargs,
        )
        contents = payload["contents"]
        generation_config = payload["generation_config"]

        response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
        )
        return response

    @staticmethod
    def parse_logprobs(
        outputs: Union[Dict, List[Dict]],
        tokens: List[List[int]] = None,
        ctxlens: List[int] = None,
        **kwargs,
    ) -> List[Tuple[float, bool]]:
        raise NotImplementedError(
            "Loglikelihood is not supported for Gemini completions."
        )

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            res.append(out["candidates"][0]["content"]["parts"][0]["text"])
        return res

    @cached_property
    def api_key(self):
        key = os.environ.get("GEMINI_API_KEY", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the `GEMINI_API_KEY` environment variable."
            )
        return key

    def loglikelihood(self, requests, **kwargs):
        raise NotImplementedError(
            "Loglikelihood is not supported for Gemini completions."
        )

    def chat_template(self, chat_template: Union[bool, str] = False) -> Optional[str]:
        return ""


@register_model("gemini-chat-completions")
class GeminiChatCompletion(GeminiCompletionsAPI):
    def __init__(
        self,
        tokenizer_backend: str = None,
        tokenized_requests: bool = False,
        **kwargs,
    ):
        eval_logger.warning(
            "chat-completions endpoint requires the `--apply_chat_template` flag."
        )
        super().__init__(
            base_url=None,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            **kwargs,
        )
        if self._batch_size > 1:
            eval_logger.warning(
                "Chat completions does not support batching. Defaulting to batch size 1."
            )
            self._batch_size = 1

    def _create_payload(
        self,
        messages: List[Dict],
        generate=False,
        gen_kwargs: dict = None,
        seed=1234,
        eos=None,
        **kwargs,
    ) -> dict:
        assert type(messages) is not str, (
            "chat-completions require the --apply_chat_template flag."
        )
        gen_kwargs.pop("do_sample", False)
        if "max_tokens" in gen_kwargs:
            max_output_tokens = gen_kwargs.pop("max_tokens")
        else:
            max_output_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
        temperature = gen_kwargs.pop("temperature", 0)
        stop_sequences = handle_stop_sequences(gen_kwargs.pop("until", None), eos)
        if not isinstance(stop_sequences, (list, tuple)):
            stop_sequences = [stop_sequences]

        generation_config = {
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
            "stop_sequences": stop_sequences[:4],
            "candidate_count": 1, # Gemini always returns 1 candidate for now
            **gen_kwargs,
        }
        return {
            "contents": format_gemini_contents(messages),
            "generation_config": generation_config,
        }

    def model_call(
        self,
        messages: Union[List[List[int]], List[str], List[JsonChatStr]],
        *,
        generate: bool = True,
        gen_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> Optional[dict]:
        gen_kwargs = copy.deepcopy(gen_kwargs)
        payload = self._create_payload(
            self.create_message(messages),
            generate=generate,
            gen_kwargs=gen_kwargs,
            seed=self._seed,
            eos=self.eos_string,
            **kwargs,
        )
        contents = payload["contents"]
        generation_config = payload["generation_config"]

        thinking_budget = generation_config.pop("thinking_budget", 0)
        if '2.5-flash' in self.model:
            config = genai.types.GenerateContentConfig(
                    **generation_config,
                    thinking_config=genai.types.ThinkingConfig(thinking_budget=thinking_budget) # Disables thinking
                )
        else:
            config = genai.types.GenerateContentConfig(**generation_config)
        # print("Generation config:", config)
        response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config
        )
        return response

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        res = []
        try:
            if not isinstance(outputs, list):
                outputs = [outputs]
            for out in outputs:
                res.append(out.candidates[0].content.parts[0].text)
        except Exception as e:
            print(f"Error parsing generations: {e}")
            res.append("")
        return res

    def tok_encode(
        self,
        string: Union[str, Any],
        left_truncate_len=None,
        add_special_tokens=None,
        **kwargs,
    ) -> Union[List[str], List[int], Any]:
        return string

    def loglikelihood(self, requests, **kwargs):
        raise NotImplementedError(
            "Loglikelihood is not supported for chat completions. Consider using the completions API instead."
        )
    
def format_gemini_contents(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts a list of messages from a common format to the format required
    by the Google Generative AI Python SDK.

    The function transforms a structure like:
    [{'role': 'user', 'content': 'Hello'}]
    into:
    [{'role': 'user', 'parts': [{'text': 'Hello'}]}]

    Args:
        contents: A list of message dictionaries, where each dictionary
                is expected to have 'role' and 'content' keys.

    Returns:
        A list of message dictionaries in the format expected by the
        generate_content method.
    """
    formatted_contents = []
    for message in contents:
        # Ensure the message has the keys we need to avoid errors
        if 'role' not in message or 'content' not in message:
            # You might want to raise an error or log a warning here
            print(f"Skipping invalid message: {message}")
            continue

        formatted_contents.append({
            "role": message["role"],
            "parts": [{"text": message["content"]}]
        })
    return formatted_contents

