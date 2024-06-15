import abc
import asyncio
import os
from functools import cached_property
from typing import Any, List, Literal, Optional, Tuple, Union

import aiohttp
import requests
from aiohttp import ClientSession
from tqdm.asyncio import tqdm_asyncio

from lm_eval.api.model import TemplateLM, eval_logger
from lm_eval.api.registry import register_model
from lm_eval.models.utils import Collator, chunks, handle_pad_token


@register_model("test")
class TemplateAPI(TemplateLM):
    def __init__(
        self,
        model: str = None,
        pretrained: str = None,
        base_url: str = None,
        tokenizer: Optional[str] = None,
        tokenizer_backend: Optional[
            Literal["tiktoken", "huggingface", None]
        ] = "huggingface",
        truncate: bool = False,
        concurrent=1,
        max_gen_toks: int = 256,
        batch_size: Union[str, int] = 1,
        seed: int = 1234,
        max_length: Optional[int] = 2058,
        add_bos_token: bool = False,
        custom_prefix_token_id=None,
    ) -> None:
        super().__init__()
        self.model = model or pretrained
        self.base_url = base_url
        self._tokenizer = tokenizer
        if "auto" in batch_size:
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
            eval_logger.warning(
                "Concurrent requests are disabled. To enable concurrent requests, set `concurrent > 1`."
            )
        self._concurrent = concurrent
        self.tokenizer_backend = tokenizer_backend
        self.add_bos_token = add_bos_token
        self.custom_prefix_token_id = custom_prefix_token_id

        if self.tokenizer_backend is None:
            self.tokenizer = None
        else:
            if self.tokenizer_backend == "huggingface":
                import transformers  # noqa: E401

                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    self._tokenizer if self._tokenizer else self.model
                )
                handle_pad_token(self.tokenizer)
            elif self.tokenizer_backend == "tiktoken":
                try:
                    import tiktoken

                    self.tokenizer = tiktoken.encoding_for_model(self.model)
                except ModuleNotFoundError:
                    raise Exception(
                        "attempted to use 'openai' LM type, but package`tiktoken` is not installed. \
            please install these via `pip install lm-eval[api]` or `pip install -e .\"[api]\"`",
                    )
                if self.base_url:
                    eval_logger.warning(
                        f"Passed `base_url={self.base_url}` but using (OpenAI) Tiktoken tokenizer backend. "
                        "Pass `tokenizer_backend=huggingface` and provide the HF tokenizer name if your model does not use Tiktoken."
                    )

    @abc.abstractmethod
    def _create_payload(
        self, messages, generate=True, gen_kwargs: dict = None, **kwargs
    ) -> dict:
        """This method is responsible for creating the json payload that will be sent to the API."""
        raise NotImplementedError

    @abc.abstractmethod
    def parse_logprobs(
        self,
        outputs: Union[Any, List[Any]],
        tokens: List[List[int]] = None,
        ctxlen: List[int] = None,
        **kwargs,
    ) -> Tuple[float, bool]:
        raise NotImplementedError

    @abc.abstractmethod
    def parse_generations(
        self, outputs: Union[Any, List[Any]], contexts: str, **kwargs
    ) -> str:
        raise NotImplementedError

    def loglikelihood_rolling(self, requests, **kwargs):
        return ""

    @cached_property
    def api_key(self):
        return ""

    @cached_property
    def header(self):
        return {"Authorization": f"Bearer {self.api_key}"}

    @cached_property
    def eot_token_id(self) -> Optional[int]:
        if self.tokenizer is None:
            return None
        else:
            if self.tokenizer_backend == "huggingface":
                return self.tokenizer.eos_token_id
            elif self.tokenizer_backend == "tiktoken":
                return self.tokenizer.eot_token_id

    @cached_property
    def prefix_token_id(self) -> Optional[int]:
        if self.tokenizer is None:
            return None
        else:
            if self.custom_prefix_token_id is not None:
                return self.custom_prefix_token_id
            if self.tokenizer.bos_token_id is not None:
                return self.tokenizer.bos_token_id
            return self.tokenizer.eos_token_id

    def tok_encode(
        self,
        string: str,
        left_truncate_len=None,
        add_special_tokens=None,
        **kwargs,
    ) -> Union[List[str], List[int]]:
        if self.tokenizer is None:
            encoding = [string]
            return encoding
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

            return encoding.input_ids

        else:
            # if not isinstance(string, (list, tuple)):
            #     string = [string]
            encoding = self.tokenizer.encode(string)
            return encoding

    # @retry(
    #     stop=stop_after_attempt(5),
    #     wait=wait_exponential(multiplier=1, min=2, max=10),
    #     reraise=True,
    # )
    def model_call(
        self,
        messages: List[Union[List[int], List[str], List[dict]]],
        generate=True,
        **kwargs,
    ) -> Optional[dict]:
        if isinstance(messages[0][0], str):
            messages = [m for sublist in messages for m in sublist]
        try:
            response = requests.post(
                self.base_url,
                json=self._create_payload(messages, generate=generate, **kwargs),
                headers=self.header,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(e)
            return None

    # @retry(
    #     stop=stop_after_attempt(5),
    #     wait=wait_exponential(multiplier=1, min=2, max=10),
    #     reraise=True,
    # )
    async def amodel_call(
        self,
        session: ClientSession,
        messages: List[Union[List[int], List[str], List[dict]]],
        generate=True,
        **kwargs,
    ) -> Optional[dict]:
        # if messages are strings then we need List[str, ...] for the json payload
        if isinstance(messages[0][0], str):
            messages = [m for sublist in messages for m in sublist]
        # try:
        payload = self._create_payload(messages, generate=generate, **kwargs)
        async with session.post(
            self.base_url,
            json=payload,
            headers=self.header,
        ) as response:
            # response.raise_for_status()
            return await response.json()
        # except RetryError:
        #     return None

    def batch_logliklehood_requests(
        self, chunk: List[Tuple[Tuple[str, str], List[int], List[int]]]
    ) -> Tuple[List[List[int]], List[int], List[Tuple[str, str]]]:
        inputs = []
        ctxlens = []
        cache_keys = []
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
        self, requests: List, generate: bool = True, **kwargs
    ):
        conn = aiohttp.TCPConnector(limit=self._concurrent)
        async with ClientSession(connector=conn) as session:
            tasks = [
                asyncio.create_task(
                    self.amodel_call(session, message, generate=generate, **kwargs)
                )
                for message in chunks(requests, n=self._batch_size)
            ]
            return await tqdm_asyncio.gather(*tasks, desc="Requesting API")

    def _loglikelihood_tokens(self, requests, **kwargs):
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
        if self._concurrent == 1:
            for chunk in chunked:
                inputs, ctxlens, cache_keys = self.batch_logliklehood_requests(chunk)
                outputs = self.model_call(messages=inputs, generate=False)
                if isinstance(outputs, dict):
                    outputs = [outputs]
                for answer, cache_key in zip(
                    self.parse_logprobs(outputs=outputs, tokens=inputs, ctxlen=ctxlens),
                    cache_keys,
                ):
                    res.append(answer)
                    # partial caching
                    if cache_key is not None:
                        self.cache_hook.add_partial("loglikelihood", cache_key, answer)
        else:
            inputs = [self.batch_logliklehood_requests(chunk) for chunk in chunked]
            inputs, ctxlens, cache_keys = zip(*inputs)
            inputs, ctxlens, cache_keys = inputs[0], ctxlens[0], cache_keys[0]
            outputs = asyncio.run(self.get_batched_requests(inputs, generate=False))
            answers = self.parse_logprobs(
                outputs=outputs,
                tokens=inputs,
                ctxlens=ctxlens,
            )

            for answer_, cached in zip(answers, cache_keys):
                res.append(answer_)
                # partial caching
                if cached is not None:
                    self.cache_hook.add_partial("loglikelihood", cached, answer_)

        return re_ord.get_original(res)

    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        res = []

        def _collate_gen(_requests):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            return -len(_requests[0][1]), _requests[0][0]

        # batch tokenize contexts
        context, all_gen_kwargs = zip(*(req.args for req in requests))
        encodings_list = [self.tok_encode(ctx) for ctx in context]
        requests = [
            ((a, b), c) for a, b, c in zip(context, encodings_list, all_gen_kwargs)
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
            for chunk in chunked:
                context_and_encoding, all_gen_kwargs = zip(*chunk)
                context, context_encoding = zip(*context_and_encoding)
                outputs = self.model_call(
                    messages=context_encoding,
                    generate=True,
                    gen_kwargs=all_gen_kwargs[0],
                )
                for generated_text, context in zip(
                    self.parse_generations(
                        outputs=outputs,
                        contexts=context,
                    ),
                    context_encoding,
                ):
                    res.append(generated_text)

                    # partial caching
                    if context is not None:
                        self.cache_hook.add_partial(
                            "generate_until",
                            (context, all_gen_kwargs[0]),
                            generated_text,
                        )
        else:
            for chunk in chunked:
                context_and_encoding, all_gen_kwargs = zip(*chunk)
                context, context_encoding = zip(*context_and_encoding)
                outputs = asyncio.run(
                    self.get_batched_requests(
                        context_encoding, generate=True, gen_kwargs=all_gen_kwargs[0]
                    )
                )
                # generated_texts = [
                #     self.parse_generations(outputs=out, contexts=ctx)
                #     for out, ctx in zip(outputs, context)
                # ]
                for gen_text, cached in zip(
                    self.parse_generations(outputs, contexts=context), context
                ):
                    res.append(gen_text)
                    # partial caching
                    if cached is not None:
                        self.cache_hook.add_partial(
                            "generate_until", (cached, all_gen_kwargs[0]), gen_text
                        )

        return re_ord.get_original(res)


@register_model("openai-completions", "local-completions")
class OpenAICompletionsAPI(TemplateAPI):
    def __init__(self, **kwargs):
        base_url = "https://api.openai.com/v1/completions"
        tokenizer_backend = "tiktoken"
        super().__init__(
            base_url=base_url, tokenizer_backend=tokenizer_backend, **kwargs
        )

    def _create_payload(
        self, messages, generate=False, gen_kwargs: dict = None, **kwargs
    ) -> dict:
        if generate:
            gen_kwargs.pop("do_sample", False)
            max_tokens = gen_kwargs.pop("max_tokens", self._max_gen_toks)
            temperature = gen_kwargs.pop("temperature", 0)
            stop = gen_kwargs.pop("until", "<|endoftext|>")
            return {
                "prompt": messages,
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": stop,
                **gen_kwargs,
            }
        else:
            return {
                "model": self.model,
                "prompt": messages,
                "max_tokens": 1,
                "logprobs": 2,
                "echo": True,
            }

    def parse_logprobs(
        self,
        outputs: Union[Any, List[Any]],
        tokens: List[List[int]] = None,
        ctxlens: List[int] = None,
        **kwargs,
    ) -> List[Tuple[float, bool]]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            for choice, ctxlen in zip(out["choices"], ctxlens):
                assert ctxlen > 0, "Context length must be greater than 0"
                logprobs = sum(choice["logprobs"]["token_logprobs"][ctxlen:-1])
                tokens = choice["logprobs"]["token_logprobs"][ctxlen:-1]
                top_logprobs = choice["logprobs"]["top_logprobs"][ctxlen:-1]
                is_greedy = True
                for tok, top in zip(tokens, top_logprobs):
                    if tok != max(top, key=top.get):
                        is_greedy = False
                        break
                res.append((logprobs, is_greedy))
        return res

    def parse_generations(
        self, outputs: Union[Any, List[Any]], contexts: List[str], **kwargs
    ) -> List[str]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            for choices in out["choices"]:
                res.append(choices["text"])
        return res

    @cached_property
    def api_key(self):
        return os.environ.get("OPENAI_API_KEY", "")
