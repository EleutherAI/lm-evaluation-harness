import os
from typing import Dict, List, Tuple, Union

from tqdm import tqdm

from lm_eval import utils
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import retry_on_specific_exceptions


eval_logger = utils.eval_logger

model_context = {
    "GigaChat": 8192,
    "GigaChat-Plus": 32768,
    "GigaChat Pro": 8192,
}  # GigaChat:latest-???


def gigachat_completion(
    client,  #: gigachat.GigaChat,
    model: str,
    prompt: str,
    max_tokens_to_sample: int,
    temperature: float,
    until: List[str],
    chat_history: Union[List[Dict[str, str]], List],
    **kwargs,
) -> str:
    """Wrapper function around the GigaChat API client with exponential back-off
    in case of RateLimitError.
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
        chat_history: Union[List[Dict[str, str]], List]
            Either messages from request if apply_chat_template is True or empty list otherwise
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
    except ModuleNotFoundError:
        raise Exception(
            "attempted to use 'gigachat' LM type, but package `gigachat` is not installed. \
please install gigachat via `pip install lm-eval[gigachat]` or `pip install -e .[gigachat]`",
        )

    messages = []
    if chat_history:
        for message in chat_history:
            messages.append(
                gigachat.models.Messages(
                    role=message["role"],
                    content=message["content"],
                )
            )
    else:
        eval_logger.warning(
            "You are trying to use GigaChat without chat_template. It may lead to inappropriate model behavior. \
                Please, set `--apply_chat_template` and `--system_instruction`  arguments."
        )
        messages.append(
            gigachat.models.Messages(
                role=gigachat.models.MessagesRole.USER,
                content=prompt,
            )
        )

    def _exception_callback(e: Exception, sleep_time: float = 10) -> None:
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
            GigaChat model, possible values: [GigaChat, GigaChat:latest, GigaChat-Plus, GigaChat-Pro]
        :param max_tokens_to_sample: int
            Maximum number of tokens to sample from the model
        :param temperature: float
            Sampling temperature. Cannot be set to zero!
        :param kwargs: Any
            Additional model_args to pass to the API client.
        """
        super().__init__()

        try:
            import gigachat
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'gigachat' LM type, but package `gigachat` is not installed. \
please install gigachat via `pip install lm-eval[gigachat]` or `pip install -e .[gigachat]`",
            )

        self.model = model
        self.chat_history = []
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
        return model_context[self.model]

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
                "attempted to use 'gigachat' LM type, but package `gigachat` is not installed. \
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
                    chat_history=self.chat_history,
                    **self.kwargs,
                )
                res.append(response)

                self.cache_hook.add_partial("generate_until", request, response)
            except (
                gigachat.exceptions.AuthenticationError,
                gigachat.exceptions.ResponseError,
            ) as e:
                eval_logger.critical(
                    f"""API error {e.args[1]}: {e.args[2].decode('utf8').split('"message":')[-1][:-1]}"""
                )
                break
        return res

    def apply_chat_template(self, chat_history: List[Dict[str, str]]) -> str:
        """
        Apply chat template in the gigachat_completion func using gigachat library
        We do not have access to gigachat tokenizer. This is our solution:
        Set chat_history as an attribute and pass it to chat completion func.
        Return a list as a string to avoid raising errors.
        """
        self.chat_history = chat_history
        return str(chat_history)

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
        generation = generation[: min(stop_idxs)]
    return generation
