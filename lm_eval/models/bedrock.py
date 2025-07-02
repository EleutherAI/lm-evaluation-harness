import json
from importlib.util import find_spec
from typing import Any, List, Tuple

from tqdm import tqdm

from lm_eval import utils
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


eval_logger = utils.eval_logger


def bedrock_chat(
    client,  # boto3.client("bedrock-runtime")
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    stop: List[str],
    **kwargs: Any,
) -> str:
    """Wrapper function around the Bedrock chat completion API client with exponential back-off
    in case of RateLimitError.

    params:
        client: boto3.Client
            Bedrock runtime client
        model: str
            Bedrock model e.g. 'anthropic.claude-3-haiku-20240307-v1:0'
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

    if not find_spec("boto3"):
        raise Exception(
            "attempted to use 'bedrock' LM type, but package `boto3` is not installed. \
please install boto3 via `pip install 'lm-eval[bedrock]'` or `pip install -e '.[bedrock]'`",
        )

    def _exception_callback(e: Exception, sleep_time: float) -> None:
        eval_logger.warning(
            f"RateLimitError occurred: {e.__cause__}\n Retrying in {sleep_time} seconds"
        )

    # @retry_on_specific_exceptions(
    #     on_exceptions=[
    #         client.ThrottlingException,
    #         client.ModelTimeoutException,
    #         client.InternalServerException,
    #     ],
    #     max_retries=None,  # retry forever, consider changing
    #     on_exception_callback=_exception_callback,
    # )
    def messages():
        # structured payload for request

        formatted_messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
        body = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": formatted_messages,
        }

        # update body with params then encode
        body.update(kwargs)
        body_bytes = json.dumps(body).encode("utf-8")

        response = client.invoke_model(
            body=body_bytes,
            contentType="application/json",
            accept="application/json",
            modelId=model,
        )
        response_body = json.loads(response.get("body").read())

        return response_body["content"][0]["text"]

    return messages()


@register_model("bedrock")
class BedrockChatLM(LM):
    def __init__(
        self,
        model: str,
        batch_size: int = 1,
        max_tokens: int = 256,
        temperature: float = 0,  # defaults to 1
        **kwargs,  # top_p, top_k, etc.
    ) -> None:
        """Bedrock API wrapper.

        :param model: str
            Bedrock model e.g. 'anthropic.claude-3-haiku-20240307-v1:0'
        :param max_tokens: int
            Maximum number of tokens to sample from the model
        :param temperature: float
            Sampling temperature
        :param kwargs: Any
            Additional model_args to pass to the API client
        """
        super().__init__()

        try:
            import boto3
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'bedrock' LM type, but package `boto3` is not installed. \
please install boto3 via `pip install 'lm-eval[bedrock]'` or `pip install -e '.[boto3]'`",
            )

        self.model = model
        # defaults to using os.environ.get("AWS_REGION"), os.environ.get("AWS_ACCESS_KEY_ID"),
        # os.environ.get("AWS_SECRET_ACCESS_KEY")
        self.client = boto3.client("bedrock-runtime")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tokenizer = None  # no tokenizer available
        self.kwargs = kwargs

    @property
    def eot_token_id(self):
        # Not sure but anthropic.HUMAN_PROMPT ?
        raise NotImplementedError("Tokenizer not available.")

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
        return self.tokenizer.encode(string).ids

    def tok_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("No support for logits.")

    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        if not find_spec("boto3"):
            raise Exception(
                "attempted to use 'bedrock' LM type, but package `boto3` is not installed. \
please install boto3 via `pip install 'lm-eval[bedrock]'` or `pip install -e '.[bedrock]'`",
            )
        import botocore

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
                response = bedrock_chat(
                    client=self.client,
                    model=self.model,
                    prompt=inp,
                    max_tokens=max_gen_toks,
                    temperature=temperature,  # TODO: implement non-greedy sampling for bedrock
                    stop=until,  # type: ignore
                    **self.kwargs,
                )
                res.append(response)

                self.cache_hook.add_partial("generate_until", request, response)

            except botocore.exceptions.ClientError as error:  # type: ignore # noqa: F821
                eval_logger.critical(f"Boto client error: {error}")
                break

            except botocore.exceptions.ParamValidationError as error:  # type: ignore # noqa: F821
                eval_logger.critical(
                    f"The parameters you provided are incorrect: {error}"
                )
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
