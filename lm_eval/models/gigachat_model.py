import os
from typing import Any, List, Tuple, Dict, Union

from tqdm import tqdm

from lm_eval import utils
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import retry_on_specific_exceptions


eval_logger = utils.eval_logger


def gigachat_completion(
    client,  #: gigachat.GigaChat,
    model: str,
    prompt: str,
    max_tokens_to_sample: int,
    temperature: float,
    until: List[str],
    user_end: str,
    **kwargs,
) -> str:
    """Wrapper function around the GigaChat API client with exponential back-off
    in case of RateLimitError.
    params:
        client: gigachat.GigaChat
            GigaChat API client
        model: str
            GigaChat model e.g. 'GigaChat-Pro', 'GigaChat'
        prompt: str
            Prompt to feed to the model
        max_tokens: int
            Maximum number of tokens to sample from the model
        temperature: float
            Sampling temperature
        until: List[str]
            List of stop-words
        user_end: str
            Additional instructions from yaml task config. Add at the end of the prompt.
        kwargs: Any
            Additional model_args to pass to the API client. May be:
            profanity check: bool, turn onn censor. Default: False
            top_p: float, nucleus params
            repetition_penalty: float, repetition_penalty.
            do_sample: sampling type.
            For more: https://developers.sber.ru/docs/ru/gigachat/api/reference/rest/post-chat
    """
    try:
        import gigachat
        import ast
    except ModuleNotFoundError:
        raise Exception(
            "attempted to use 'gigachat' LM type, but package `gigachat` or `ast` are not installed. \
please install gigachat via `pip install lm-eval[gigachat]` or `pip install -e .[gigachat]`",
        )

    prompt = ast.literal_eval(prompt)
    messages = []
    if isinstance(prompt, list):
        for num, message in enumerate(prompt):
            if num == (len(message) - 1):
                message["content"] += user_end
            messages.append(
                gigachat.models.Messages(
                    role=message["role"],
                    content=message["content"],
                )
            )
    elif isinstance(prompt, str):
        messages.append(
            gigachat.models.Messages(
                role=gigachat.models.MessagesRole.USER,
                content=prompt + user_end,
            )
        )
    else:
        raise TypeError("Unknown input type")

    def _exception_callback(e: Exception, sleep_time: float = 10) -> None:
        """
        ReadTimeout - if there is no responce from the server.
        """
        eval_logger.warning(
            f"GigaChatError occurred: {e.__cause__}\n Retrying in {sleep_time} seconds"
        )

    @retry_on_specific_exceptions(
        on_exceptions=[
            gigachat.exceptions.GigaChatException,
            gigachat.exceptions.ResponseError,
            gigachat.exceptions.AuthenticationError,
        ],
        max_retries=5,
        on_exception_callback=_exception_callback,
    )
    def completion():

        payload = gigachat.models.Chat(
            messages=messages,
            model=model,
            max_tokens=max_tokens_to_sample,
            temperature=temperature,
            **kwargs,
        )
        response = client.chat(payload).choices[0].message.content

        if until:
            # TBD: cut for stream generation
            response = cut_generation(response, until)

        return response

    return completion()


@register_model("gigachat_llms")
class GigaChatLM(LM):

    def __init__(
        self,
        model: str = "GigaChat",
        max_tokens: int = 256,
        temperature: float = 1e-10,
        **kwargs,  # top_p,  etc.
    ) -> None:
        """GigaChat API wrapper.

        :param model: str
            GC model e.g. 'GigaChat', 'GigaChar-Pro'
        :param max_tokens_to_sample: int
            Maximum number of tokens to sample from the model
        :param temperature: float
            Sampling temperature
        :param kwargs: Any
            Additional model_args to pass to the API client
        """
        super().__init__()

        try:
            import gigachat
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'gigachat' LM type, but package `gigachat` or `ast` are not installed. \
please install gigachat via `pip install lm-eval[gigachat]` or `pip install -e .[gigachat]`",
            )

        self.model = model
        self.client = gigachat.GigaChat(
            credentials=os.environ.get("GIGACHAT_API_KEY"),
            scope="GIGACHAT_API_CORP",
            verify_ssl_certs=False,
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs

    @property
    def eot_token_id(self):
        raise NotImplementedError("No idea about gc tokenization.")

    @property
    def max_length(self) -> int:
        return 2048

    @property
    def max_gen_toks(self) -> int:
        return self.max_tokens

    @property
    def batch_size(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError("No support for logits.")

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError("No support for logits.")

    def tok_encode(self, string: str) -> List[int]:
        return NotImplementedError("No idea about gc tokenization.")

    def tok_decode(self, tokens: List[int]) -> str:
        return NotImplementedError("No idea about gc tokenization.")

    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("No support for logits.")

    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        try:
            import gigachat
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'gigachat' LM type, but package `gigachat` or `ast` are not installed. \
please install gigachat via `pip install lm-eval[gigachat]` or `pip install -e .[gigachat]`",
            )

        if not requests:
            return []

        _requests: List[Tuple[str, dict]] = [req.args for req in requests]

        res = []
        for request in tqdm(_requests, disable=disable_tqdm):
            try:
                inp = request[0]
                request_args = request[1]
                until = request_args.get("until", None)
                if isinstance(until, str):
                    until = [until]
                # generation_kwargs
                user_end = request_args.get("user_end", "")
                max_gen_toks = request_args.get("max_gen_toks", self.max_length)
                temperature = request_args.get("temperature", self.temperature)
                if temperature == 0:
                    temperature = 1e-10
                response = gigachat_completion(
                    client=self.client,
                    model=self.model,
                    prompt=inp,
                    max_tokens_to_sample=max_gen_toks,
                    temperature=temperature,
                    until=until,
                    user_end=user_end,
                    **self.kwargs,
                )
                res.append(response)

                self.cache_hook.add_partial("generate_until", request, response)
            except gigachat.exceptions.AuthenticationError as e:
                eval_logger.critical(
                    f"""API error {e.args[1]}: {e.args[2].decode('utf8').split('"message":')[-1][:-1]}"""
                )
                break
            except gigachat.exceptions.ResponseError as e:
                eval_logger.critical(
                    f"""API error {e.args[1]}: {e.args[2].decode('utf8').split('"message":')[-1][:-1]}"""
                )
                break

        return res

    def apply_chat_template(self, chat_history: List[Dict[str, str]]) -> str:
        """
        Apply chat template in the gigachat_completion func using gigachat library
        We do not have acess to gigachat tokenizer.
        Return Dict as str and then use it as a Dict again to process in through gigachat meassages.
        """
        return str(chat_history)

    @property
    def tokenizer_name(self) -> str:
        """
        Apply chat template in the gigachat_completion func using gigachat library.
        We do not have acess to gigachat tokenizer.
        Return gigachat_tokenizer as a name.
        """
        return str("gigachat_tokenizer")

    @property
    def chat_template(self) -> str:
        """
        Apply chat template in the gigachat_completion func using gigachat library.
        We do not have acess to gigachat tokenizer.
        """
        return str("No idea about gc tokenization.")

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


def cut_generation(generation, stop):
    """
    GigaChat API has no stop argument.
    Use this func in order to cut gc generation.
    TBD: async -> stop_generation
    """

    stop_idxs = [generation.find(sub) for sub in stop if generation.find(sub) != -1]
    if stop_idxs:
        return generation[: min(stop_idxs)]
    else:
        return generation
