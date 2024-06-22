import abc
import asyncio
import copy
import itertools
import json
from collections import namedtuple
from functools import cached_property
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
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

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM, eval_logger
from lm_eval.models.utils import Collator, chunks, handle_pad_token


JsonChatStr = namedtuple("JsonChatStr", ["prompt"])


class TemplateAPI(TemplateLM):
    def __init__(
        self,
        model: str = None,
        pretrained: str = None,
        base_url: str = None,
        tokenizer: Optional[str] = None,
        # Logliklehood tasks require a tokenizer to calculate context lengths,
        # however the requests can be sent as a string if the API doesn't support token inputs.
        # use tokenized_requests=False
        tokenizer_backend: Optional[
            Literal["tiktoken", "huggingface", None]
        ] = "huggingface",
        truncate: bool = False,
        # concurrent requests. More useful if not batching
        concurrent=1,
        max_gen_toks: int = 256,
        batch_size: Union[str, int] = 1,
        seed: int = 1234,
        max_length: Optional[int] = 2058,
        add_bos_token: bool = False,
        custom_prefix_token_id=None,
        # send the requests as tokens or strings
        tokenized_requests=True,
    ) -> None:
        try:
            pass
        except Exception:
            raise Exception(
                "Attempted to use an API model, but the required packages are not installed. "
                'Please install these via `pip install lm-eval[api]` or `pip install -e ."[api]"`'
            )

        super().__init__()
        self.model = model or pretrained
        self.base_url = base_url
        self.tokenizer = tokenizer
        if not isinstance(batch_size, int) and "auto" in batch_size:
            eval_logger.warning(
                "Automatic batch size is not supported for API models. Defaulting to batch size 1."
            )
        elif int(batch_size) > 1:
            eval_logger.warning(
                "Using batch size > 1. Be sure that your API supports batched requests with arbitrary (total) sequence lengths."
            )
        self._batch_size = int(batch_size) if batch_size != "auto" else 1
        self._truncate = truncate
        self._max_gen_toks = max_gen_toks
        self._seed = seed
        self.max_length = max_length
        if concurrent <= 1:
            eval_logger.info(
                "Concurrent requests are disabled. To enable concurrent requests, set `concurrent > 1`."
            )
        self._concurrent = concurrent
        self.tokenizer_backend = tokenizer_backend
        self.add_bos_token = add_bos_token
        self.custom_prefix_token_id = custom_prefix_token_id
        self.tokenized_requests = tokenized_requests

        if self.tokenizer_backend is None:
            self.tokenizer = None
            self.tokenized_requests = False
        else:
            if self.tokenizer_backend == "huggingface":
                import transformers

                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    self.tokenizer if self.tokenizer else self.model
                )
                self.tokenizer = handle_pad_token(self.tokenizer)
            elif self.tokenizer_backend == "tiktoken":
                try:
                    import tiktoken

                    self.tokenizer = tiktoken.encoding_for_model(self.model)
                except ModuleNotFoundError:
                    raise Exception(
                        "attempted to use 'openai' LM type, but package`tiktoken` is not installed. \
            please install these via `pip install lm-eval[api]` or `pip install -e .\"[api]\"`",
                    )
                if self.base_url and self.tokenizer_backend == "tiktoken":
                    eval_logger.warning(
                        f"Passed `base_url={self.base_url}` but using (OpenAI) Tiktoken tokenizer backend. "
                        "Pass `tokenizer_backend=huggingface` and provide the HF tokenizer name if your model does not use Tiktoken."
                    )

    @abc.abstractmethod
    def _create_payload(
        self, messages, *, generate=True, gen_kwargs: dict = None, **kwargs
    ) -> dict:
        """This method is responsible for creating the json payload that will be sent to the API."""
        raise NotImplementedError

    def create_message(
        self,
        messages: Union[List[List[int]], List[str], List[JsonChatStr]],
        generate=False,
    ) -> Union[List[List[int]], List[dict], List[str], str]:
        """Helper method to"""

        if isinstance(messages[0], JsonChatStr):
            # for chat completions we need to decode the json string to list[dict,...]
            assert (
                self._batch_size == 1
            ), "non-tokenized chat requests are only supported with batch_size=1"
            return json.loads(messages[0].prompt)

        if not self.tokenized_requests:
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

    @abc.abstractmethod
    def parse_logprobs(
        self,
        outputs: Union[Any, List[Any]],
        tokens: List[List[int]] = None,
        ctxlen: List[int] = None,
        **kwargs,
    ) -> List[Tuple[float, bool]]:
        """Method used to parse the logprobs from the (optionally batched) API response. This method should return a list of tuples"""
        raise NotImplementedError

    @abc.abstractmethod
    def parse_generations(self, outputs: Union[Any, List[Any]], **kwargs) -> List[str]:
        """Method used to parse the generations from the (optionally batched) API response. This method should return a list of str"""
        raise NotImplementedError

    @cached_property
    def api_key(self):
        """Override this property to return the API key for the API request."""
        return ""

    @cached_property
    def header(self):
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
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        if self.tokenizer_backend == "huggingface":
            return self.tokenizer.apply_chat_template(
                chat_history, tokenize=False, add_generation_prompt=True
            )
        else:
            # bit of a hack. We'll re-encode back before sending to the API
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
            if self.tokenizer_backend == "huggingface":
                if self.custom_prefix_token_id is not None:
                    return self.custom_prefix_token_id
                if self.tokenizer.bos_token_id is not None:
                    return self.tokenizer.bos_token_id
                return self.tokenizer.eos_token_id
            else:
                return self.tokenizer.eot_token

    def tok_encode(
        self,
        string: str,
        left_truncate_len=None,
        add_special_tokens=None,
        **kwargs,
    ) -> Union[List[int], str]:
        if self.tokenizer_backend is None:
            return string
        elif self.tokenizer_backend == "huggingface":
            # by default for CausalLM - false or self.add_bos_token is set
            if add_special_tokens is None:
                add_special_tokens = False or self.add_bos_token
            # otherwise the method explicitly defines the value
            else:
                add_special_tokens = add_special_tokens

            encoding = self.tokenizer.encode(
                string, add_special_tokens=add_special_tokens, **kwargs
            )

            # left-truncate the encoded context to be at most `left_truncate_len` tokens long
            if left_truncate_len:
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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def model_call(
        self,
        messages: Union[List[List[int]], List[str], List[JsonChatStr]],
        *,
        generate: bool = True,
        **kwargs,
    ) -> Optional[dict]:
        try:
            response = requests.post(
                self.base_url,
                json=self._create_payload(
                    self.create_message(messages), generate=generate, **kwargs
                ),
                headers=self.header,
            )
            response.raise_for_status()
            return response.json()
        except RetryError:
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def amodel_call(
        self,
        session: ClientSession,
        messages: List[Union[List[int], str]],
        *,
        cache_keys=None,
        ctxlens: Optional[List[int]] = None,
        generate: bool = True,
        **kwargs,
    ) -> Optional[List[Union[str, Tuple[float, bool]]]]:
        payload = self._create_payload(
            self.create_message(messages), generate=generate, **kwargs
        )
        try:
            async with session.post(
                self.base_url,
                json=payload,
                headers=self.header,
            ) as response:
                response.raise_for_status()
                outputs = await response.json()
                if generate:
                    answers = self.parse_generations(
                        outputs=outputs,
                    )
                    for res, cache in zip(answers, cache_keys):
                        self.cache_hook.add_partial(
                            "generate_until",
                            cache,
                            res,
                        )
                    return answers
                else:
                    answers = self.parse_logprobs(
                        outputs=outputs,
                        tokens=messages,
                        ctxlens=ctxlens,
                    )
                    for res, cache in zip(answers, cache_keys):
                        self.cache_hook.add_partial("loglikelihood", cache, res)

                    return answers
        except RetryError:
            return None

    def batch_logliklehood_requests(
        self, chunks: Iterable[List[Tuple[Tuple[str, str], List[int], List[int]]]]
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
        requests: List,
        cache_keys,
        *,
        generate: bool = True,
        ctxlens=None,
        **kwargs,
    ):
        conn = TCPConnector(limit=self._concurrent)
        async with ClientSession(connector=conn) as session:
            tasks = [
                asyncio.create_task(
                    self.amodel_call(
                        session,
                        message,
                        cache_keys=cache_keys,
                        generate=generate,
                        ctxlens=ctxlens,
                        **kwargs,
                    )
                )
                for message in chunks(requests, n=self._batch_size)
            ]
            return await tqdm_asyncio.gather(*tasks, desc="Requesting API")

    def _loglikelihood_tokens(self, requests, **kwargs):
        assert (
            self.tokenizer is not None
        ), "Tokenizer is required for loglikelihood tasks to compute context lengths"
        res = []
        assert (
            self.tokenizer is not None
        ), "Tokenizer is required for loglikelihood tasks"

        def _collate(req: Tuple[Tuple[str, str], List[int], List[int]]):
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
        chunked = re_ord.get_batched(n=self._batch_size if self._concurrent <= 1 else 0)
        if self._concurrent <= 1:
            pbar = tqdm(desc="Requesting API", total=len(requests))
            for chunk in chunked:
                inputs, ctxlens, cache_keys = self.batch_logliklehood_requests([chunk])
                outputs = self.model_call(
                    messages=self.create_message(inputs), generate=False
                )
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
                        # partial caching
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
            encodings_list = self.tok_encode(requests)
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
                if self.tokenized_requests:
                    req = encodings_list
                else:
                    req = contexts
                outputs = self.model_call(
                    messages=req,
                    generate=True,
                    gen_kwargs=all_gen_kwargs[0],
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
                if self.tokenized_requests:
                    req = encodings_list
                else:
                    req = contexts
                res = itertools.chain.from_iterable(
                    asyncio.run(
                        self.get_batched_requests(
                            req,
                            generate=True,
                            cache_keys=[
                                (ctx, gen_k)
                                for ctx, gen_k in zip(contexts, all_gen_kwargs)
                            ],
                            gen_kwargs=copy.deepcopy(all_gen_kwargs[0]),
                        )
                    )
                )

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
                        prefix_token=self.eot_token_id,
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
        return loglikelihoods
