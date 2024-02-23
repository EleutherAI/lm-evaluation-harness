import asyncio  # for running API calls concurrently
import atexit
import copy
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import os
import re  # for matching endpoint from request URL
import signal
import tempfile
import time  # for sleeping after rate limit is hit
from collections import defaultdict

# Openai parallel processor imports
from dataclasses import (
    dataclass,
    field,
)

# for storing API inputs, outputs, and metadata
from importlib.util import find_spec
from typing import List, Literal, Optional, Tuple

import aiohttp  # for making API calls concurrently
import tiktoken  # for counting tokens
from tqdm import tqdm

import lm_eval.models.utils
from lm_eval import utils
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import retry_on_specific_exceptions
from lm_eval.utils import eval_logger


async def process_api_requests_from_file(
    requests_filepath: str,
    save_filepath: str,
    request_url: str,
    api_key: str,
    max_requests_per_minute: float,
    max_tokens_per_minute: float,
    token_encoding_name: str,
    max_attempts: int,
    logging_level: int,
):
    """Processes API requests in parallel, throttling to stay under rate limits."""
    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = (
        0.001  # 1 ms limits max throughput to 1,000 requests per second
    )

    # initialize logging
    logging.basicConfig(level=logging_level)
    logging.debug(f"Logging initialized at level {logging_level}")

    # infer API endpoint and construct request header
    api_endpoint = api_endpoint_from_url(request_url)
    request_header = {"Authorization": f"Bearer {api_key}"}
    # use api-key header for Azure deployments
    if "/deployments" in request_url:
        request_header = {"api-key": f"{api_key}"}

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = (
        task_id_generator_function()
    )  # generates integer IDs of 0, 1, 2, ...
    status_tracker = (
        StatusTracker()
    )  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # initialize flags
    file_not_finished = True  # after file is empty, we'll skip reading it
    logging.debug("Initialization complete.")

    # initialize file reading
    with open(requests_filepath) as file:
        # `requests` will provide requests one at a time
        requests = file.__iter__()
        logging.debug("File opened. Entering main loop")
        async with aiohttp.ClientSession() as session:  # Initialize ClientSession here
            while True:
                # get next request (if one is not already waiting for capacity)
                if next_request is None:
                    if not queue_of_requests_to_retry.empty():
                        next_request = queue_of_requests_to_retry.get_nowait()
                        logging.debug(
                            f"Retrying request {next_request.task_id}: {next_request}"
                        )
                    elif file_not_finished:
                        try:
                            # get new request
                            request_json = json.loads(next(requests))
                            next_request = APIRequest(
                                task_id=next(task_id_generator),
                                request_json=request_json,
                                token_consumption=num_tokens_consumed_from_request(
                                    request_json, api_endpoint, token_encoding_name
                                ),
                                attempts_left=max_attempts,
                                metadata=request_json.pop("metadata", None),
                            )
                            status_tracker.num_tasks_started += 1
                            status_tracker.num_tasks_in_progress += 1
                            logging.debug(
                                f"Reading request {next_request.task_id}: {next_request}"
                            )
                        except StopIteration:
                            # if file runs out, set flag to stop reading it
                            logging.debug("Read file exhausted")
                            file_not_finished = False

                # update available capacity
                current_time = time.time()
                seconds_since_update = current_time - last_update_time
                available_request_capacity = min(
                    available_request_capacity
                    + max_requests_per_minute * seconds_since_update / 60.0,
                    max_requests_per_minute,
                )
                available_token_capacity = min(
                    available_token_capacity
                    + max_tokens_per_minute * seconds_since_update / 60.0,
                    max_tokens_per_minute,
                )
                last_update_time = current_time

                # if enough capacity available, call API
                if next_request:
                    next_request_tokens = next_request.token_consumption
                    if (
                        available_request_capacity >= 1
                        and available_token_capacity >= next_request_tokens
                    ):
                        # update counters
                        available_request_capacity -= 1
                        available_token_capacity -= next_request_tokens
                        next_request.attempts_left -= 1

                        # call API
                        asyncio.create_task(
                            next_request.call_api(
                                session=session,
                                request_url=request_url,
                                request_header=request_header,
                                retry_queue=queue_of_requests_to_retry,
                                save_filepath=save_filepath,
                                status_tracker=status_tracker,
                            )
                        )
                        next_request = None  # reset next_request to empty

                # if all tasks are finished, break
                if status_tracker.num_tasks_in_progress == 0:
                    break

                # main loop sleeps briefly so concurrent tasks can run
                await asyncio.sleep(seconds_to_sleep_each_loop)

                # if a rate limit error was hit recently, pause to cool down
                seconds_since_rate_limit_error = (
                    time.time() - status_tracker.time_of_last_rate_limit_error
                )
                if (
                    seconds_since_rate_limit_error
                    < seconds_to_pause_after_rate_limit_error
                ):
                    remaining_seconds_to_pause = (
                        seconds_to_pause_after_rate_limit_error
                        - seconds_since_rate_limit_error
                    )
                    await asyncio.sleep(remaining_seconds_to_pause)
                    # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
                    logging.warn(
                        f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}"
                    )

        # after finishing, log final status
        logging.info(
            f"""Parallel processing complete. Results saved to {save_filepath}"""
        )
        if status_tracker.num_tasks_failed > 0:
            logging.warning(
                f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {save_filepath}."
            )
        if status_tracker.num_rate_limit_errors > 0:
            logging.warning(
                f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate."
            )


# dataclasses


@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    metadata: dict
    result: list = field(default_factory=list)

    async def call_api(
        self,
        session: aiohttp.ClientSession,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
    ):
        """Calls the OpenAI API and saves results."""
        logging.info(f"Starting request #{self.task_id}")
        error = None
        try:
            async with session.post(
                url=request_url, headers=request_header, json=self.request_json
            ) as response:
                response = await response.json()
            if "error" in response:
                logging.warning(
                    f"Request {self.task_id} failed with error {response['error']}"
                )
                status_tracker.num_api_errors += 1
                error = response
                if "Rate limit" in response["error"].get("message", ""):
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= (
                        1  # rate limit errors are counted separately
                    )

        except Exception as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e
        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(
                    f"Request {self.request_json} failed after all attempts. Saving errors: {self.result}"
                )
                data = (
                    [self.request_json, [str(e) for e in self.result], self.metadata]
                    if self.metadata
                    else [self.request_json, [str(e) for e in self.result]]
                )
                append_to_jsonl(data, save_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            data = (
                [self.request_json, response, self.metadata]
                if self.metadata
                else [self.request_json, response]
            )
            append_to_jsonl(data, save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            logging.debug(f"Request {self.task_id} saved to {save_filepath}")


# functions


def api_endpoint_from_url(request_url):
    """Extract the API endpoint from the request URL."""
    match = re.search("^https://[^/]+/v\\d+/(.+)$", request_url)
    if match is None:
        # for Azure OpenAI deployment urls
        match = re.search(
            r"^https://[^/]+/openai/deployments/[^/]+/(.+?)(\?|$)", request_url
        )
    return match[1]


def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")


def num_tokens_consumed_from_request(
    request_json: dict,
    api_endpoint: str,
    token_encoding_name: str,
):
    """Count the number of tokens in the request. Only supports completion and embedding requests."""
    encoding = tiktoken.get_encoding(token_encoding_name)
    # if completions request, tokens = prompt + n * max_tokens
    if api_endpoint.endswith("completions"):
        max_tokens = request_json.get("max_tokens", 15)
        n = request_json.get("n", 1)
        completion_tokens = n * max_tokens

        # chat completions
        if api_endpoint.startswith("chat/"):
            num_tokens = 0
            for message in request_json["messages"]:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens -= 1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens + completion_tokens
        # normal completions
        else:
            prompt = request_json["prompt"]
            if isinstance(prompt, str):  # single prompt
                prompt_tokens = len(encoding.encode(prompt))
                num_tokens = prompt_tokens + completion_tokens
                return num_tokens
            elif isinstance(prompt, list):  # multiple prompts
                prompt_tokens = sum([len(encoding.encode(p)) for p in prompt])
                num_tokens = prompt_tokens + completion_tokens * len(prompt)
                return num_tokens
            else:
                raise TypeError(
                    'Expecting either string or list of strings for "prompt" field in completion request'
                )
    # if embeddings request, tokens = input tokens
    elif api_endpoint == "embeddings":
        input = request_json["input"]
        if isinstance(input, str):  # single input
            num_tokens = len(encoding.encode(input))
            return num_tokens
        elif isinstance(input, list):  # multiple inputs
            num_tokens = sum([len(encoding.encode(i)) for i in input])
            return num_tokens
        else:
            raise TypeError(
                'Expecting either string or list of strings for "inputs" field in embedding request'
            )
    # more logic needed to support other API calls (e.g., edits, inserts, DALL-E)
    else:
        raise NotImplementedError(
            f'API endpoint "{api_endpoint}" not implemented in this script'
        )


def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1


def process_chat_request(messages, model, idx, **kwargs):
    request = {
        "messages": messages,
        "model": model,
        "metadata": {"idx": idx},
    }

    request.update(kwargs)

    return json.dumps(request)


def get_result(response, ctxlen: int) -> Tuple[float, bool]:
    """Process results from OpenAI API response.

    :param response: dict
        OpenAI API Response
    :param ctxlen: int
        Length of context (so we can slice them away and only keep the predictions)
    :return:
        continuation_logprobs: np.array
            Log probabilities of continuation tokens
        is_greedy: bool
            whether argmax matches given continuation exactly
    """
    is_greedy = True
    logprobs = response.logprobs.token_logprobs
    continuation_logprobs = sum(logprobs[ctxlen:])

    for i in range(ctxlen, len(response.logprobs.token_logprobs)):
        token = response.logprobs.token_logprobs[i]
        top_tokens = response.logprobs.top_logprobs[i]
        top_token = max(top_tokens.keys(), key=lambda x: top_tokens[x])
        if top_token != token:
            is_greedy = False
            break

    return continuation_logprobs, is_greedy


def oa_completion(client, chat: bool = False, **kwargs):
    """Query OpenAI API for completion.

    Retry with back-off until they respond
    """
    if not find_spec("openai") or not find_spec("tiktoken"):
        raise Exception(
            "attempted to use 'openai' LM type, but package `openai` or `tiktoken` are not installed. "
            "Please install these via `pip install lm-eval[openai]` or `pip install -e .[openai]`"
        )
    else:
        import openai

    def _exception_callback(e: Exception, sleep_time: float) -> None:
        import traceback

        traceback.print_exc()

    @retry_on_specific_exceptions(
        on_exceptions=[openai.OpenAIError],
        max_retries=None,  # retry forever, consider changing
        on_exception_callback=_exception_callback,
    )
    def completion():
        if chat:
            return client.chat.completions.create(**kwargs)
        else:
            return client.completions.create(**kwargs)

    return completion()


@register_model("openai-completions", "local-completions")
class OpenaiCompletionsLM(LM):
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        model: str,
        base_url: str = None,
        tokenizer: Optional[str] = None,
        tokenizer_backend: Literal["tiktoken", "huggingface"] = "tiktoken",
        truncate: bool = False,
        max_gen_toks: int = 256,
        batch_size: int = 1,
        seed: int = 1234,
        max_length: Optional[int] = None,
    ) -> None:
        """

        :param engine: str
            OpenAI API engine (e.g. gpt-3.5-turbo-instruct)
        :param truncate: bool
            Truncate input if too long (if False and input is too long, throw error)
        """
        super().__init__()
        self.seed = seed
        try:
            import openai  # noqa: E401
            import tiktoken
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'openai' LM type, but package `openai` or `tiktoken` are not installed. \
    please install these via `pip install lm-eval[openai]` or `pip install -e .[openai]`",
            )
        self.model = model
        self.base_url = base_url
        self.tokenizer_backend = tokenizer_backend
        self.truncate = truncate
        self._batch_size = batch_size
        self._max_gen_toks = max_gen_toks
        self._max_length = max_length

        # if we have a local model, use HF tokenizer over tiktoken
        if self.tokenizer_backend == "huggingface":
            import transformers  # noqa: E401

            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                tokenizer if tokenizer else self.model
            )
            self.vocab_size = self.tokenizer.vocab
            self.end_of_text_token_id = self.tokenizer.eos_token
        elif self.tokenizer_backend == "tiktoken":
            if self.base_url:
                eval_logger.warning(
                    f"Passed `base_url={self.base_url}` but using Tiktoken tokenizer backend. "
                    "Pass `tokenizer_backend=huggingface` and provide the HF tokenizer name if your model does not use Tiktoken."
                )

            self.tokenizer = tiktoken.encoding_for_model(self.model)
            self.vocab_size = self.tokenizer.n_vocab
            self.end_of_text_token_id = self.tokenizer.eot_token
        else:
            raise ValueError(
                f"Expected tokenizer_backend to be one of ['tiktoken', 'huggingface'] but got {self.tokenizer_backend}"
            )

        # Read from environment variable OPENAI_API_KEY
        # Set to EMPTY for local
        openai.api_key = os.environ["OPENAI_API_KEY"]
        if self.base_url:
            self.client = openai.OpenAI(base_url=self.base_url)
        else:
            self.client = openai.OpenAI()

    @property
    def eot_token_id(self):
        return self.end_of_text_token_id

    @property
    def max_length(self) -> int:
        if self._max_length:
            return self._max_length
        else:
            return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def tok_encode(self, string: str) -> List[int]:
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    def _encode_pair(
        self, context: str, continuation: str
    ) -> Tuple[List[int], List[int]]:
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole_enc = self.tok_encode(context + continuation)
        context_enc = self.tok_encode(context)
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc

    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        new_reqs = []
        for context, continuation in [req.args for req in requests]:
            if context == "":
                # end of text as context
                context_enc, continuation_enc = (
                    [self.eot_token_id],
                    self.tok_encode(continuation),
                )
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def _loglikelihood_tokens(
        self, requests, disable_tqdm: bool = False
    ) -> List[Tuple[float, bool]]:
        res = []

        def _collate(x):
            # this doesn't efficiently handle last-token differences yet, but those are kinda annoying because
            # it's not guaranteed that the 100 or so logprobs we get to see actually contain all the continuations
            # we care about, and so we need some kind of backup for when it isn't
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        re_ord = utils.Reorderer(requests, _collate)

        for chunk in tqdm(
            list(lm_eval.models.utils.chunks(re_ord.get_reordered(), self.batch_size)),
            disable=disable_tqdm,
        ):
            inps = []
            ctxlens = []
            for cache_key, context_enc, continuation_enc in chunk:
                # max_length+1 because the API takes up to 2049 tokens, including the first context token
                inp = (context_enc + continuation_enc)[-(self.max_length + 1) :]
                # TODO: the logic is much simpler if we just look at the length of continuation tokens
                ctxlen = len(context_enc) - max(
                    0, len(context_enc) + len(continuation_enc) - (self.max_length + 1)
                )

                inps.append(inp)
                ctxlens.append(ctxlen)

            response = oa_completion(
                client=self.client,
                model=self.model,
                prompt=inps,
                echo=True,
                max_tokens=0,
                temperature=0.0,
                logprobs=10,
                seed=self.seed,
            )

            for resp, ctxlen, (cache_key, context_enc, continuation_enc) in zip(
                response.choices, ctxlens, chunk
            ):
                answer = get_result(resp, ctxlen)

                res.append(answer)

                # partial caching
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)
        return re_ord.get_original(res)

    def generate_until(self, requests) -> List[str]:
        if not requests:
            return []
        res = []
        requests = [req.args for req in requests]

        def _collate(x):
            toks = self.tok_encode(x[0])
            return len(toks), x[0]

        re_ord = utils.Reorderer(requests, _collate)

        def sameuntil_chunks(xs, size):
            ret = []
            lastuntil = xs[0][1]
            for x in xs:
                if len(ret) >= size or x[1] != lastuntil:
                    yield ret, lastuntil
                    ret = []
                    lastuntil = x[1]
                ret.append(x)

            if ret:
                yield ret, lastuntil

        # todo: more intelligent batching for heterogeneous `until`
        for chunk, request_args in tqdm(
            list(sameuntil_chunks(re_ord.get_reordered(), self.batch_size))
        ):
            inps = []
            self._max_gen_toks = request_args.pop("max_gen_toks", self.max_gen_toks)
            for context, _ in chunk:
                context_enc = self.tok_encode(context)
                inp = context_enc[-(self.max_length - self.max_gen_toks) :]
                inps.append(inp)

            until = request_args.pop("until", ["<|endoftext|>"])
            request_args.pop("do_sample", None)
            request_args["temperature"] = request_args.get("temperature", 0)

            response = oa_completion(
                client=self.client,
                model=self.model,
                prompt=inps,
                max_tokens=self.max_gen_toks,
                stop=until,
                seed=self.seed,
                **request_args,
            )
            for resp, (context, args_) in zip(response.choices, chunk):
                s = getattr(resp, "text")

                until_ = until

                for term in until_:
                    if len(term) > 0:
                        s = s.split(term)[0]

                # partial caching
                self.cache_hook.add_partial(
                    "generate_until", (context, {"until": until_}), s
                )

                res.append(s)
        return re_ord.get_original(res)

    def _model_call(self, inps):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override generate_until
        raise NotImplementedError()

    def loglikelihood_rolling(self, requests) -> List[float]:
        loglikelihoods = []

        for (string,) in tqdm([req.args for req in requests]):
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


@register_model("openai-chat-completions", "local-chat-completions")
class OpenaiChatCompletionsLM(LM):
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",  # GPT model or Local model using HuggingFace model paths
        base_url: str = None,
        truncate: bool = False,
        **kwargs,
    ) -> None:
        """

        :param model: str
            Implements an OpenAI-style chat completion API for
            accessing both OpenAI OR locally-hosted models using
            HuggingFace Tokenizer
            OpenAI API model (e.g. gpt-3.5-turbo)
            using the **gen_kwargs passed on init
        :param truncate: bool
            Truncate input if too long (if False and input is too long, throw error)
        """
        super().__init__()
        try:
            import openai  # noqa: E401
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'openai' LM type, but package `openai` or `tiktoken` are not installed. \
    please install these via `pip install lm-eval[openai]` or `pip install -e .[openai]`",
            )
        self.model = model
        self.base_url = base_url
        self.truncate = truncate

        # Read from environment variable OPENAI_API_KEY
        # Set to EMPTY for local
        if self.base_url:
            self.client = openai.OpenAI(base_url=self.base_url)
        else:
            self.client = openai.OpenAI()  # openai.AsyncOpenAI()

    @property
    def max_length(self) -> int:
        # Note: the OpenAI API supports up to 2049 tokens, with the first token being the first input token
        return 2048

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def generate_until(self, requests) -> List[str]:
        if self.base_url:
            res = defaultdict(list)
            re_ords = {}

            # we group requests by their generation_kwargs,
            # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
            # in the same batch.
            grouper = lm_eval.models.utils.Grouper(requests, lambda x: str(x.args[1]))
            for key, reqs in grouper.get_grouped().items():
                # within each set of reqs for given kwargs, we reorder by token length, descending.
                re_ords[key] = utils.Reorderer(
                    [req.args for req in reqs], lambda x: (-len(x[0]), x[0])
                )

            pbar = tqdm(total=len(requests), disable=(self.rank != 0))
            for key, re_ord in re_ords.items():
                # n needs to be 1 because messages in
                # chat completion are not batch but
                # is regarded as a single conversation.
                chunks = lm_eval.models.utils.chunks(re_ord.get_reordered(), n=1)
                for chunk in chunks:
                    contexts, all_gen_kwargs = zip(*chunk)
                    inps = [
                        {"role": "user", "content": context} for context in contexts
                    ]

                    gen_kwargs = all_gen_kwargs[0]
                    until = None
                    if isinstance(kwargs := copy.deepcopy(gen_kwargs), dict):
                        if "do_sample" in kwargs.keys():
                            kwargs.pop("do_sample")
                        if "until" in kwargs.keys():
                            until = kwargs.pop("until")
                            if isinstance(until, str):
                                until = [kwargs]
                            elif not isinstance(until, list):
                                raise ValueError(
                                    f"Expected repr(kwargs['until']) to be of type Union[str, list] but got {until}"
                                )
                            kwargs["stop"] = until
                        kwargs["max_tokens"] = kwargs.pop(
                            "max_gen_toks", self.max_gen_toks
                        )
                    else:
                        raise ValueError(
                            f"Expected repr(kwargs) to be of type repr(dict) but got {kwargs}"
                        )

                    response = oa_completion(
                        client=self.client,
                        chat=True,
                        messages=inps,
                        model=self.model,
                        **kwargs,
                    )

                    for resp, (context, args_) in zip(response.choices, chunk):
                        s = resp.message.content

                        if until is not None:
                            for term in until:
                                if len(term) > 0:
                                    s = s.split(term)[0]

                        res[key].append(s)

                        self.cache_hook.add_partial(
                            "generate_until", (context, {"until": until}), s
                        )
                        pbar.update(1)
                # reorder this group of results back to original unsorted form
                res[key] = re_ord.get_original(res[key])

            pbar.close()

            final_responses = grouper.get_original(res)

        else:
            # Dummy request to get headers with rate limits
            openai_headers = self.client.chat.completions.with_raw_response.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": "Respond with a full stop only (.):"}
                ],
                stop=["."],
                max_tokens=1,
            ).headers

            temp_dir = tempfile.gettempdir()
            requests_file_path = os.path.join(
                temp_dir, "lm_eval_harness_requests.jsonl"
            )
            responses_file_path = os.path.join(
                temp_dir, "lm_eval_harness_responses.jsonl"
            )

            def clean_up_requests(signum=None, frame=None):
                if os.path.exists(requests_file_path):
                    os.remove(requests_file_path)
                    print("Cached requests temp file deleted.")
                if os.path.exists(responses_file_path):
                    os.remove(responses_file_path)
                    print("Cached responses temp file deleted.")
                if signum is not None:
                    exit(0)  # Only exit if called by a signal

            atexit.register(clean_up_requests)
            signal.signal(signal.SIGINT, clean_up_requests)
            signal.signal(signal.SIGTERM, clean_up_requests)

            pbar = tqdm(total=len(requests), disable=(self.rank != 0))
            with open(requests_file_path, "w") as file:
                for idx, request in enumerate(requests):
                    context = request.arguments[0]
                    gen_kwargs = request.arguments[1]

                    inps = [{"role": "user", "content": context}]

                    until = None
                    if isinstance(kwargs := copy.deepcopy(gen_kwargs), dict):
                        if "do_sample" in kwargs.keys():
                            kwargs.pop("do_sample")
                        if "until" in kwargs.keys():
                            until = kwargs.pop("until")
                            if isinstance(until, str):
                                until = [kwargs]
                            elif not isinstance(until, list):
                                raise ValueError(
                                    f"Expected repr(kwargs['until']) to be of type Union[str, list] but got {until}"
                                )
                            kwargs["stop"] = until
                        if "max_requests_per_minute" in kwargs.keys():
                            max_requests_per_minute = kwargs.pop(
                                "max_requests_per_minute"
                            )
                        else:
                            # Can't currently get RPM from openai headers, only RPD, have taken up with OpenAI
                            max_requests_per_minute = 500  # Default from tier 1 rate limits (lowest paid limits)
                        if "max_tokens_per_minute" in kwargs.keys():
                            max_tokens_per_minute = kwargs.pop("max_tokens_per_minute")
                        else:  # Get TPM from users account
                            max_tokens_per_minute = int(
                                openai_headers.get("x-ratelimit-limit-tokens")
                            )
                        if "max_attempts" in kwargs.keys():
                            max_attempts = kwargs.pop("max_attempts")
                        else:
                            max_attempts = 10
                        kwargs["max_tokens"] = kwargs.pop(
                            "max_gen_toks", self.max_gen_toks
                        )
                    else:
                        raise ValueError(
                            f"Expected repr(kwargs) to be of type repr(dict) but got {kwargs}"
                        )

                    request_to_cache = process_chat_request(
                        messages=inps,
                        model=self.model,
                        idx=idx,
                        **kwargs,
                    )
                    file.write(request_to_cache + "\n")

                print(f"Requests written to {requests_file_path}.")

            print(f"Max requests per minute: {max_requests_per_minute}")
            print(f"Max tokens per minute: {max_tokens_per_minute}")
            print(
                "use --gen_kwargs max_requests_per_minute=N,max_tokens_per_minute=N to override"
            )

            asyncio.run(
                process_api_requests_from_file(
                    requests_filepath=str(requests_file_path),
                    save_filepath=str(responses_file_path),
                    request_url=str("https://api.openai.com/v1/chat/completions"),
                    api_key=str(os.environ["OPENAI_API_KEY"]),
                    max_requests_per_minute=float(
                        max_requests_per_minute * 0.75
                    ),  # *0.75 to leave some headroom
                    max_tokens_per_minute=float(max_tokens_per_minute * 0.75),
                    token_encoding_name="cl100k_base",
                    max_attempts=int(max_attempts),
                    logging_level=int(30),
                )
            )

            with open(responses_file_path, "r") as responses_temp_file:
                lines = responses_temp_file.readlines()

            results = []
            for line in lines:
                response_object = json.loads(line)
                context = response_object[0]["messages"][0]["content"]
                response = response_object[1]["choices"]
                idx = response_object[2]["idx"]

                for resp in response:
                    s = resp["message"]["content"]

                    if until is not None:
                        for term in until:
                            if len(term) > 0:
                                s = s.split(term)[0]

                    results.append((idx, s))

                    self.cache_hook.add_partial(
                        "generate_until", (context, {"until": until}), s
                    )
                    pbar.update(1)

            results.sort(key=lambda x: x[0])

            pbar.close()
            clean_up_requests()
            final_responses = [s for idx, s in results]

        return final_responses

    def loglikelihood(self, requests):
        raise NotImplementedError("No support for logits.")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("No support for logits.")
