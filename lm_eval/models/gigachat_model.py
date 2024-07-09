import logging
import os
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from lm_eval import utils
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import retry_on_specific_exceptions


eval_logger = utils.eval_logger


logging.getLogger("httpx").setLevel(
    logging.WARNING
)  # turn off logging 200 status for each iteration


def gigachat_completion(
    client,  #: gigachat.GigaChat,
    model: str,
    prompt: str,
    max_tokens_to_sample: int,
    temperature: float,
    until: List[str],
    chat_template_is_on: bool,
    **kwargs,
) -> str:
    """Wrapper function around the GigaChat API client with exponential back-off
    in case of RateLimitError.
    For authorization set environmental variables "GIGACHAT_CREDENTIALS" and "GIGACHAT_SCOPE" for your API auth_data and scope (GIGACHAT_API_CORP or GIGACHAT_API_PERS) respectively.
    Skip sample after 5 retries if there is an error with GigaChat API occurred.
    params:
        client: gigachat.GigaChat
            GigaChat API client
        model: str
            GigaChat model, possible values: [GigaChat, GigaChat:latest, GigaChat-Plus, GigaChat-Pro]
        prompt: str
            Prompt to feed to the model
        max_tokens: int
            Maximum number of tokens to sample from the model
        temperature: float
            Sampling temperature
        until: List[str]
            List of stop-words
        chat_template_is_on: bool
            Use chat_template or not
        kwargs: Any
            Additional model_args to pass to the API client. May be:
            profanity check: bool, censor status. Default: True
            top_p: float, nucleus params. The default value depends on the selected model and may change with model updates
            repetition_penalty: float, repetition_penalty. The default value depends on the selected model and may change with model updates
            n: int, the number of response options to be generated for each input message. Possible values: [1; 4]. Default: 1
            stream: bool, specifies that messages should be sent in parts in the stream. Default: False
    """
    try:
        import gigachat
        import httpx
    except ModuleNotFoundError:
        raise Exception(
            "attempted to use 'gigachat' LM type, but packages `gigachat` or `httpx` are not installed. \
please install gigachat via `pip install lm-eval[gigachat]` or `pip install -e .[gigachat]`",
        )

    messages = []
    if not chat_template_is_on:
        messages.append(
            gigachat.models.Messages(
                role=gigachat.models.MessagesRole.USER,
                content=prompt,
            )
        )
    else:
        seq = prompt.split("<role>")[1:]
        for message in seq:
            role, content = message.split("<content>")
            messages.append(
                gigachat.models.Messages(
                    role=role,
                    content=content,
                )
            )

    def _exception_callback(e: Exception, sleep_time: float = 10) -> None:
        eval_logger.warning(
            f"GigaChatError occurred: {e.__str__()}\n Retrying in {sleep_time} seconds"
        )

    @retry_on_specific_exceptions(
        on_exceptions=[
            httpx.ReadTimeout,  # it is like a RateLimitError
            httpx.ConnectTimeout,
        ],
        max_retries=None,
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
            response = cut_generation(response, until)
        if not response:
            response = " "  # avoid None in resps
        return response

    return completion()


@register_model("gigachat_llms")
class GigaChatLM(LM):
    def __init__(
        self,
        model: str = "GigaChat",
        max_tokens: Optional[
            int
        ] = None,  # default is None as API will automatically choose the most optimal value
        temperature: Optional[float] = None,
        scope: str = "GIGACHAT_API_CORP",
        verify_ssl_certs: bool = False,
        **kwargs,  # top_p,  etc.
    ) -> None:
        """GigaChat API wrapper.

        :param model: str
            GigaChat model, possible values: [GigaChat, GigaChat:latest, GigaChat-Plus, GigaChat-Pro]
        :param max_tokens_to_sample: int
            Maximum number of tokens to sample from the model
        :param temperature: float
            Sampling temperature. Cannot be set to zero!
        :param scope: str
            If you have a personal token, set this parameter to "GIGACHAT_API_PERS"; otherwise, use the default
        :param verify_ssl_certs: bool
            Set this parameter if you have your certificates installed to ensure greater security
        :param kwargs: Any
            Additional model_args to pass to the API client.
        """
        super().__init__()

        try:
            import gigachat
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'gigachat' LM type, but packages `gigachat` or `httpx` are not installed. \
please install gigachat via `pip install lm-eval[gigachat]` or `pip install -e .[gigachat]`",
            )

        self.model = model
        self.client = gigachat.GigaChat(
            credentials=os.environ.get("GIGACHAT_CREDENTIALS"),
            scope=os.environ.get("GIGACHAT_SCOPE", scope),
            verify_ssl_certs=verify_ssl_certs,
            timeout=100,
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        self.chat_template_is_used = False

    @property
    def eot_token_id(self):
        raise NotImplementedError("No idea about GigaChat tokenization.")

    @property
    def max_length(self) -> int:
        return None

    @property
    def max_gen_toks(self) -> int:
        """
        Set max_gen_toks to None as API itself defines max token limit for each model type.
        """
        return None

    @property
    def batch_size(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError("No support for logits.")

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError("No support for logits.")

    def tok_encode(self, string: str) -> List[int]:
        return NotImplementedError("No idea about GigaChat tokenization.")

    def tok_decode(self, tokens: List[int]) -> str:
        return NotImplementedError("No idea about GigaChat tokenization.")

    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("No support for logits.")

    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        try:
            import gigachat
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'gigachat' LM type, but packages `gigachat` or `httpx` are not installed. \
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
                until = request_args.get("until")
                if isinstance(until, str):
                    until = [until]
                # generation_kwargs
                max_gen_toks = request_args.get("max_gen_toks", None)
                temperature = request_args.get("temperature", self.temperature)

                if (
                    "do_sample" in self.kwargs.keys()
                ):  # API does not have do sample option.
                    if not self.kwargs[
                        "do_sample"
                    ]:  # Ensure greedy decoding if do_sample=False
                        self.kwargs["repetition_penalty"] = 1
                        self.kwargs["top_p"] = 0
                    elif temperature == 0:
                        eval_logger.warning(
                            "You cannot set do_sample=True and temperature=0. Automatically setting temperature=1."
                        )
                        temperature = 1.0

                if (
                    temperature == 0
                ):  # Ensure greedy decoding by setting top_p=0 and repetition_penalty = 1
                    temperature = (
                        1.0  # temperature cannot be set to zero. Use top_p instead
                    )
                    self.kwargs["repetition_penalty"] = 1
                    self.kwargs["top_p"] = 0

                if not self.chat_template_is_used:
                    eval_logger.warning(
                        "You are trying to use GigaChat without chat_template. It may lead to inappropriate model behavior. \
                            Please, set `--apply_chat_template` and `--system_instruction`  arguments."
                    )

                response = gigachat_completion(
                    client=self.client,
                    model=self.model,
                    prompt=inp,
                    max_tokens_to_sample=max_gen_toks,
                    temperature=temperature,
                    until=until,
                    chat_template_is_on=self.chat_template_is_used,
                    **self.kwargs,
                )

                res.append(response)

                self.cache_hook.add_partial("generate_until", request, response)
            except (gigachat.exceptions.ResponseError,) as e:
                status, mes = parse_exception(e)
                eval_logger.critical(f"""API error {status}: {mes}""")
                break
        return res

    def apply_chat_template(self, chat_history: List[Dict[str, str]]) -> str:
        """
        Apply chat template in the gigachat_completion func using gigachat library
        We do not have access to gigachat tokenizer. This is our solution:
        Set chat_history as an attribute and pass it to chat completion func.
        Return a list as a string to avoid raising errors.
        """
        if not self.chat_template_is_used:
            self.chat_template_is_used = True
        prompt = ""
        for dct in chat_history:
            prompt += f"<role>{dct['role']}<content>{dct['content']}"
        return prompt

    @property
    def tokenizer_name(self) -> str:
        """
        Apply chat template in the gigachat_completion func using gigachat library.
        We do not have access to gigachat tokenizer.
        Return gigachat_tokenizer as a name.
        """
        return str("gigachat_tokenizer")

    @property
    def chat_template(self) -> str:
        """
        Apply chat template in the gigachat_completion func using gigachat library.
        We do not have access to gigachat tokenizer.
        """
        return str("No idea about GigaChat tokenization.")

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
    Use this func in order to cut GigaChat generation.
    """
    if not generation:
        generation = " "
    stop_idxs = [generation.find(sub) for sub in stop if generation.find(sub) != -1]
    if stop_idxs:
        generation = generation[: min(stop_idxs)]
    return generation


def parse_exception(exp):
    import ast

    exp_dict = ast.literal_eval(exp.args[2].decode("utf8"))
    return exp_dict.get("status", exp.args[1]), exp_dict.get("message")
