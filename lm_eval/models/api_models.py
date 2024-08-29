import abc
import asyncio
import copy
import itertools
import json
from functools import cached_property
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)


try:
    import requests
    from aiohttp import ClientSession, TCPConnector
    from tenacity import RetryError, retry, stop_after_attempt, wait_exponential
    from tqdm import tqdm
    from tqdm.asyncio import tqdm_asyncio
except ModuleNotFoundError:
    pass


from importlib.util import find_spec

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.models.utils import Collator, chunks, configure_pad_token


LogLikelihoodInputs = Tuple[Tuple[str, str], List[int], List[int]]


# utility class to keep track of json encoded chats
class JsonChatStr(NamedTuple):
    prompt: str

    def encode(self, encoding):
        return self.prompt.encode(encoding)


eval_logger = utils.eval_logger


class TemplateAPI(TemplateLM):
    def __init__(
        self,
        model: str = None,
        pretrained: str = None,  # `model` takes precedence over `pretrained` when passed.
        base_url: str = None,
        tokenizer: Optional[str] = None,
        # Logliklehood tasks require a tokenizer to calculate context lengths,
        # however the requests can be sent as a string if the API doesn't support token inputs.
        # use tokenized_requests=False
        tokenizer_backend: Optional[
            Literal["tiktoken", "huggingface", None]
        ] = "huggingface",
        truncate: bool = False,
        # number of concurrent requests. More useful if not batching
        num_concurrent: int = 1,
        max_retries: int = 3,
        max_gen_toks: int = 256,
        batch_size: Union[str, int] = 1,
        seed: int = 1234,
        max_length: Optional[int] = 2048,
        add_bos_token: bool = False,
        custom_prefix_token_id=None,
        # send the requests as tokens or strings
        tokenized_requests=True,
        **kwargs,
    ) -> None:
        super().__init__()
        missing_packages = [
            pkg
            for pkg in ["aiohttp", "tqdm", "tenacity", "requests"]
            if find_spec(pkg) is None
        ]
        if missing_packages:
            raise ModuleNotFoundError(
                f"Attempted to use an API model, but the required packages {missing_packages} are not installed. "
                'Please install these via `pip install lm-eval[api]` or `pip install -e ."[api]"`'
            )
        self.model = model or pretrained
        self.base_url = base_url
        self.tokenizer = tokenizer
        if not isinstance(batch_size, int) and "auto" in batch_size:
            eval_logger.warning(
                "Automatic batch size is not supported for API models. Defaulting to batch size 1."
            )
        elif int(batch_size) > 1:
            eval_logger.warning(
                "Batch size > 1 detected. Ensure your API supports batched requests with varying total sequence lengths."
            )
        self._batch_size = int(batch_size) if batch_size != "auto" else 1
        self._truncate = truncate
        self._max_gen_toks = int(max_gen_toks)
        self._seed = int(seed)
        self.max_length = max_length
        if int(num_concurrent) <= 1:
            eval_logger.info(
                "Concurrent requests are disabled. To enable concurrent requests, set `num_concurrent` > 1."
            )
        self._concurrent = int(num_concurrent)
        self.tokenizer_backend = tokenizer_backend
        self.add_bos_token = add_bos_token
        self.custom_prefix_token_id = custom_prefix_token_id
        self.tokenized_requests = tokenized_requests
        self.max_retries = int(max_retries)

        eval_logger.info(f"Using tokenizer {self.tokenizer_backend}")
        if self.tokenizer_backend is None:
            self.tokenizer = None
            self.tokenized_requests = False
        else:
            if self.tokenizer is None:
                if self.tokenizer_backend == "huggingface":
                    import transformers

                    self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                        self.tokenizer if self.tokenizer else self.model
                    )
                    # Not used as the API will handle padding but to mirror the behavior of the HFLM
                    self.tokenizer = configure_pad_token(self.tokenizer)
                elif self.tokenizer_backend == "tiktoken":
                    try:
                        import tiktoken

                        self.tokenizer = tiktoken.encoding_for_model(self.model)
                    except ModuleNotFoundError as e:
                        raise Exception(
                            "Attempted to use 'openai' LM type, but the package `tiktoken` is not installed. "
                            "Please install it via `pip install lm-eval[api]` or `pip install -e .[api]`."
                        ) from e
                    if "openai" not in self.base_url:
                        eval_logger.warning(
                            f"Passed `base_url={self.base_url}` but using (OpenAI) Tiktoken tokenizer backend. "
                            "Pass `tokenizer_backend=huggingface` and provide the HF tokenizer name if your model does not use Tiktoken."
                        )
            else:
                import transformers

                assert isinstance(tokenizer, str), "tokenizer must be a string"
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    tokenizer,
                )

    @abc.abstractmethod
    def _create_payload(
        self,
        messages: Union[List[List[int]], List[dict], List[str], str],
        *,
        generate: bool = True,
        gen_kwargs: Optional[dict] = None,
        seed: int = 1234,
        **kwargs,
    ) -> dict:
        """This method is responsible for creating the json payload that will be sent to the API."""
        raise NotImplementedError

    def create_message(
        self,
        messages: Union[List[List[int]], List[str], List[JsonChatStr]],
        generate=False,
    ) -> Union[List[List[int]], List[dict], List[str], str]:
        """Helper method to transform the prompt into the expected API input format. messages consist of batched requests"""
        if isinstance(messages[0], JsonChatStr):
            # for chat completions we need to decode the json string to list[dict,...]
            assert (
                self._batch_size == 1
            ), "non-tokenized chat requests are only supported with batch_size=1"
            # list[dict["role":..., "content":...],...]
            return json.loads(messages[0].prompt)

        if not self.tokenized_requests:
            # if messages are tokenized:
            if isinstance(messages[0][0], int):
                # assuming decoding is lossless. However, this is only for logliklehood requests
                # as we need to compute the context length. For generations, we don't need to tokenize.
                messages = self.decode_batch(messages)
            if self._batch_size <= 1:
                # if batch is 1 return str
                return messages[0]
            else:
                # list[str,...]
                return messages

        # list[list[int], ...]
        return messages

    @staticmethod
    @abc.abstractmethod
    def parse_logprobs(
        outputs: Union[Any, List[Any]],
        tokens: List[List[int]] = None,
        ctxlen: List[int] = None,
        **kwargs,
    ) -> List[Tuple[float, bool]]:
        """Method used to parse the logprobs from the (batched) API response. This method should return a list of tuples"""
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def parse_generations(outputs: Union[Any, List[Any]], **kwargs) -> List[str]:
        """Method used to parse the generations from the (batched) API response. This method should return a list of str"""
        raise NotImplementedError

    @cached_property
    def api_key(self) -> str:
        """Override this property to return the API key for the API request."""
        return ""

    @cached_property
    def header(self) -> dict:
        """Override this property to return the headers for the API request."""
        return {"Authorization": f"Bearer {self.api_key}"}

    @property
    def chat_template(self) -> str:
        """Must be defined for LM subclasses that implement Chat Templating.
        Should return the structure of the chat template applied to user/assistant messages.
        Only used for logging and reproducibility.
        """
        return ""

    @property
    def tokenizer_name(self) -> str:
        """Must be defined for LM subclasses which implement Chat Templating.
        Should return the name of the tokenizer or chat template used.
        Used only to properly fingerprint caches when requests are being cached with `--cache_requests`, otherwise not used.
        """
        return ""

    def apply_chat_template(
        self, chat_history: List[Dict[str, str]]
    ) -> Union[str, JsonChatStr]:
        """Applies a chat template to a list of chat history between user and model."""
        if self.tokenizer_backend == "huggingface" and self.tokenized_requests:
            return self.tokenizer.apply_chat_template(
                chat_history, tokenize=False, add_generation_prompt=True
            )
        else:
            # bit of a hack. We'll load back before sending to the API
            return JsonChatStr(json.dumps(chat_history))

    @cached_property
    def eot_token_id(self) -> Optional[int]:
        if self.tokenizer is None:
            return None
        else:
            if self.tokenizer_backend == "huggingface":
                return self.tokenizer.eos_token_id
            elif self.tokenizer_backend == "tiktoken":
                return self.tokenizer.eot_token

    @cached_property
    def prefix_token_id(self) -> Optional[int]:
        if self.tokenizer is None:
            return None
        else:
            if self.custom_prefix_token_id is not None:
                return self.custom_prefix_token_id
            if self.tokenizer_backend == "huggingface":
                if self.tokenizer.bos_token_id is not None:
                    return self.tokenizer.bos_token_id
                return self.tokenizer.eos_token_id
            else:
                return self.tokenizer.eot_token

    def tok_encode(
        self,
        string: str,
        left_truncate_len: int = None,
        add_special_tokens: bool = False,
        truncation: bool = False,
        **kwargs,
    ) -> Union[List[List[int]], List[int], List[str]]:
        if self.tokenizer_backend is None:
            return [string]
        elif self.tokenizer_backend == "huggingface":
            # by default for CausalLM - false or self.add_bos_token is set
            if not add_special_tokens:
                add_special_tokens = False or self.add_bos_token
            encoding: Union[List[List[int]], List[int]] = self.tokenizer(
                string,
                add_special_tokens=add_special_tokens,
                truncation=truncation,
                return_attention_mask=False,
            ).input_ids

            # left-truncate the encoded context to be at most `left_truncate_len` tokens long
            if left_truncate_len:
                if not isinstance(string, str):
                    encoding = [enc[-left_truncate_len:] for enc in encoding]
                else:
                    encoding = encoding[-left_truncate_len:]

            return encoding

        else:
            try:
                encoding = self.tokenizer.encode(string)
            except Exception:
                encoding = self.tokenizer.encode_batch(string)
            return encoding

    def decode_batch(self, tokens: List[List[int]]) -> List[str]:
        if self.tokenizer_backend == "huggingface":
            return self.tokenizer.batch_decode(tokens)
        elif self.tokenizer_backend == "tiktoken":
            return self.tokenizer.decode_batch(tokens)

    def model_call(
        self,
        messages: Union[List[List[int]], List[str], List[JsonChatStr]],
        *,
        generate: bool = True,
        gen_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> Optional[dict]:
        # !!! Copy: shared dict for each request, need new object !!!
        gen_kwargs = copy.deepcopy(gen_kwargs)
        try:
            response = requests.post(
                self.base_url,
                json=self._create_payload(
                    self.create_message(messages),
                    generate=generate,
                    gen_kwargs=gen_kwargs,
                    seed=self._seed,
                    **kwargs,
                ),
                headers=self.header,
            )
            if not response.ok:
                eval_logger.warning(
                    f"API request failed with error message: {response.text}. Retrying..."
                )
            response.raise_for_status()
            return response.json()
        except RetryError:
            eval_logger.error(
                "API request failed after multiple retries. Please check the API status."
            )
            return None

    async def amodel_call(
        self,
        session: ClientSession,
        messages: Union[List[List[int]], List[str], List[JsonChatStr]],
        *,
        generate: bool = True,
        cache_keys: list = None,
        ctxlens: Optional[List[int]] = None,
        gen_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> Union[List[str], List[Tuple[float, bool]], None]:
        # !!! Copy: shared dict for each request, need new object !!!
        gen_kwargs = copy.deepcopy(gen_kwargs)
        payload = self._create_payload(
            self.create_message(messages),
            generate=generate,
            gen_kwargs=gen_kwargs,
            seed=self._seed,
            **kwargs,
        )
        cache_method = "generate_until" if generate else "loglikelihood"
        try:
            async with session.post(
                self.base_url,
                json=payload,
                headers=self.header,
            ) as response:
                if not response.ok:
                    error_text = await response.text()
                    eval_logger.warning(
                        f"API request failed with error message: {error_text}. Retrying..."
                    )
                # raising exception will retry the request
                response.raise_for_status()
                outputs = await response.json()
            answers = (
                self.parse_generations(
                    outputs=outputs,
                )
                if generate
                else self.parse_logprobs(
                    outputs=outputs,
                    tokens=messages,
                    ctxlens=ctxlens,
                )
            )
            if cache_keys:
                for res, cache in zip(answers, cache_keys):
                    self.cache_hook.add_partial(cache_method, cache, res)
            return answers
        # If the retries also fail
        except RetryError:
            eval_logger.error(
                "API request failed after multiple retries. Please check the API status."
            )
            return None

    def batch_logliklehood_requests(
        self, chunks: Iterable[List[LogLikelihoodInputs]]
    ) -> Tuple[List[List[int]], List[int], List[Tuple[str, str]]]:
        inputs = []
        ctxlens = []
        cache_keys = []
        for chunk in chunks:
            for cache_key, context_enc, continuation_enc in chunk:
                inp = (context_enc + continuation_enc)[-(self.max_length) :]
                ctxlen = len(context_enc) - max(
                    0, len(context_enc) + len(continuation_enc) - (self.max_length)
                )

                inputs.append(inp)
                ctxlens.append(ctxlen)
                cache_keys.append(cache_key)
        return inputs, ctxlens, cache_keys

    async def get_batched_requests(
        self,
        requests: list,
        cache_keys: list,
        *,
        generate: bool = True,
        ctxlens: List[int] = None,
        **kwargs,
    ) -> Union[List[List[str]], List[List[Tuple[float, bool]]]]:
        ctxlens = ctxlens if ctxlens else [None] * len(requests)
        conn = TCPConnector(limit=self._concurrent)
        async with ClientSession(connector=conn) as session:
            retry_: Callable[..., Awaitable[Any]] = retry(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential(multiplier=0.5, min=1, max=10),
                reraise=True,
            )(self.amodel_call)
            # Create tasks for each batch of request
            tasks = [
                asyncio.create_task(
                    retry_(
                        session=session,
                        messages=message,
                        cache_keys=cache_key,
                        generate=generate,
                        ctxlens=ctxlen,
                        **kwargs,
                    )
                )
                for message, cache_key, ctxlen in zip(
                    chunks(requests, n=self._batch_size),
                    chunks(cache_keys, n=self._batch_size),
                    chunks(ctxlens, n=self._batch_size),
                )
            ]

            return await tqdm_asyncio.gather(*tasks, desc="Requesting API")

    def _loglikelihood_tokens(self, requests, **kwargs) -> List[Tuple[float, bool]]:
        assert (
            self.tokenizer is not None
        ), "Tokenizer is required for loglikelihood tasks to compute context lengths."
        res = []

        def _collate(req: LogLikelihoodInputs):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        re_ord = Collator(
            requests,
            sort_fn=_collate,
            group_by=None,
        )
        # if concurrent then we'll batch in the async context
        chunked = re_ord.get_batched(n=self._batch_size if self._concurrent <= 1 else 0)
        if self._concurrent <= 1:
            pbar = tqdm(desc="Requesting API", total=len(requests))
            for chunk in chunked:
                inputs, ctxlens, cache_keys = self.batch_logliklehood_requests([chunk])

                outputs = retry(
                    stop=stop_after_attempt(self.max_retries),
                    wait=wait_exponential(multiplier=0.5, min=1, max=10),
                    reraise=True,
                )(self.model_call)(messages=inputs, generate=False)
                if isinstance(outputs, dict):
                    outputs = [outputs]
                for answer_, cache_key in zip(
                    self.parse_logprobs(
                        outputs=outputs, tokens=inputs, ctxlens=ctxlens
                    ),
                    cache_keys,
                ):
                    if answer_ is not None:
                        res.append(answer_)
                        # cache requests that aren't from a loglikelihood_rolling request
                        if cache_key is not None:
                            self.cache_hook.add_partial(
                                "loglikelihood", cache_key, answer_
                            )
                        pbar.update(1)
        else:
            inputs, ctxlens, cache_keys = self.batch_logliklehood_requests(chunked)
            res = itertools.chain.from_iterable(
                asyncio.run(
                    self.get_batched_requests(
                        inputs, cache_keys, generate=False, ctxlens=ctxlens
                    )
                )
            )

        return re_ord.get_original(res)

    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        res = []

        def _collate_gen(_requests):
            # sort by the length of the non-tokenized contexts
            return -len(_requests[0])

        # Let the API deal with tokenization
        requests, all_gen_kwargs = zip(*(req.args for req in requests))
        if self.tokenized_requests:
            encodings_list = self.tok_encode(
                requests, add_special_tokens=self.add_bos_token
            )
        else:
            encodings_list = [None] * len(requests)
        requests = [
            (a, b, c) for a, b, c in zip(requests, all_gen_kwargs, encodings_list)
        ]

        re_ord = Collator(
            requests,
            sort_fn=_collate_gen,
            group_by="gen_kwargs",
        )
        chunked = re_ord.get_batched(
            n=self._batch_size if self._concurrent <= 1 else 0, batch_fn=None
        )
        if self._concurrent <= 1:
            pbar = tqdm(desc="Requesting API", total=len(requests))
            for chunk in chunked:
                contexts, all_gen_kwargs, encodings_list = zip(*chunk)
                req = encodings_list if self.tokenized_requests else contexts
                outputs = retry(
                    stop=stop_after_attempt(self.max_retries),
                    wait=wait_exponential(multiplier=0.5, min=1, max=10),
                    reraise=True,
                )(self.model_call)(
                    messages=req,
                    generate=True,
                    gen_kwargs=copy.deepcopy(all_gen_kwargs[0]),
                )
                for generated_text, context in zip(
                    self.parse_generations(
                        outputs=outputs,
                        contexts=contexts,
                    ),
                    contexts,
                ):
                    if generated_text is not None:
                        res.append(generated_text)

                        # partial caching
                        if context is not None:
                            self.cache_hook.add_partial(
                                "generate_until",
                                (context, all_gen_kwargs[0]),
                                generated_text,
                            )
                            pbar.update(1)
        else:
            for chunk in chunked:
                contexts, all_gen_kwargs, encodings_list = zip(*chunk)
                req = encodings_list if self.tokenized_requests else contexts
                results = itertools.chain.from_iterable(
                    asyncio.run(
                        self.get_batched_requests(
                            req,
                            cache_keys=[(ctx, all_gen_kwargs[0]) for ctx in contexts],
                            generate=True,
                            gen_kwargs=copy.deepcopy(all_gen_kwargs[0]),
                        )
                    )
                )
                res.extend(results)

        return re_ord.get_original(res)

    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        loglikelihoods = []

        for (string,) in tqdm([req.args for req in requests], disable=disable_tqdm):
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )

            # TODO: Right now, we pass single EOT token to the Encoder and the full context to the decoder, in seq2seq case
            rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            string_nll = self._loglikelihood_tokens(
                rolling_token_windows,
                disable_tqdm=True,
            )

            # discard is_greedy
            string_nll = [x[0] for x in string_nll]

            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)

            # cache this loglikelihood_rolling request
            self.cache_hook.add_partial("loglikelihood_rolling", (string,), string_nll)
        return loglikelihoods
