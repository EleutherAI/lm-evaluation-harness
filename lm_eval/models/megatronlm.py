import json
import re
import time
from typing import List

import requests
from tqdm import tqdm

from lm_eval import utils
from lm_eval.base import BaseLM


def get_result(logprobs, is_max_logprobs, ctxlen):
    """Process results from Megatron-LM Server API response.
    :param logprobs
        List of lists of the log probs of the tokens (for concatatenated context and continuation without the first token of
         the context), e.g. [-1.4, -0.7, -0.5, -0.9]
    :param is_max_logprobs
        List of boolean flag indicating for which token the predicted probability was equal to the maximum predicted
         probability (for concatatenated context and continuation without the first token of the context),
         e.g. [False, True, True, False]
    :param ctxlen: int
        Length of context (so we can slice them away and only keep the predictions)
    :return:
        continuation_logprobs: np.array
            Sum of the log probabilities of the continuation tokens (i.e. log probability of continuation)
        is_greedy: bool
            Whether the continuation could have been the result of greedy generation
    """
    first_continuation_logprob_index = ctxlen - 1

    if first_continuation_logprob_index < 0:
        raise Exception(
            "Can't compute log probabilities and is_greedy with empty context"
        )

    continuation_logprobs = logprobs[first_continuation_logprob_index:]

    is_greedy = True

    for i in range(first_continuation_logprob_index, len(is_max_logprobs)):
        if not is_max_logprobs[i]:
            is_greedy = False
            break

    return sum(continuation_logprobs), is_greedy


class MegatronLMClient(BaseLM):
    def __init__(self, server_url, model_name, batch_size=20, truncate=False):
        """

        :param server_url: str
            Base URL of the server (e.g. https://12.23.34.45)

        :param model_name: str
            Name of model (for sanity check). Must be a substring of the model path on the server

        :param truncate: bool
            Truncate input if too long (if False and input is too long, throw error)
        """
        super().__init__()

        self.server_url = server_url

        self.model_name = model_name

        self._batch_size = batch_size

        self.truncate = truncate

        meglm_metadata = self.megatron_metadata()
        self.vocab_size = meglm_metadata["vocab_size"]
        self._eod_token_id = meglm_metadata["eod_token_id"]
        self._max_length = meglm_metadata["max_length"]

    @property
    def eot_token_id(self):
        return self._eod_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def tok_encode(self, string: str, *, is_continuation: bool = False):
        return self.tokenizer_query([string], is_continuation=is_continuation)[0]

    def tok_decode(self, tokens):
        return self.detokenizer_query([tokens])[0]

    def tok_encode_batch(
        self, string_batch: List[str], is_continuation: bool = False
    ) -> List[List[int]]:
        return self.tokenizer_query(string_batch, is_continuation)

    def tok_decode_batch(self, tokens_batch):
        return self.detokenizer_query(tokens_batch)

    def loglikelihood(self, requests):
        new_reqs = []
        contexts = []
        continuations = []
        contexts_continuations = []

        for context, continuation in requests:
            contexts.append(context)
            continuations.append(continuation)
            contexts_continuations.append(context + continuation)

        contexts_enc = self.tok_encode_batch(contexts)
        continuations_enc = self.tok_encode_batch(continuations)
        continuations_enc_iscont = self.tok_encode_batch(
            continuations, is_continuation=True
        )
        contexts_continuations_enc = self.tok_encode_batch(contexts_continuations)

        for (
            context_enc,
            continuation_enc,
            continuation_enc_iscont,
            context_continuation_enc,
            request,
        ) in zip(
            contexts_enc,
            continuations_enc,
            continuations_enc_iscont,
            contexts_continuations_enc,
            requests,
        ):
            # Check if encoded context and continuation would return the same tokens
            # with and without removing the additional white space
            if context_enc + continuation_enc != context_continuation_enc:
                if context_enc + continuation_enc_iscont == context_continuation_enc:
                    continuation_enc = continuation_enc_iscont
                else:
                    print(
                        f"WARNING: Unnatural tokenization of concatenated context ...{repr(context[-20:])} and continuation {repr(continuation)}"
                    )

            if len(context_enc) == 0:
                context_enc = [self.eot_token_id]

            new_reqs.append((request, context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        res = []

        def _collate(x):
            # this doesn't efficiently handle last-token differences yet, but those are kinda annoying because
            # it's not guaranteed that the 100 or so logprobs we get to see actually contain all the continuations
            # we care about and so we need some kind of backup for when it isn't
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        re_ord = utils.Reorderer(requests, _collate)

        for chunk in tqdm(
            list(utils.chunks(re_ord.get_reordered(), self.batch_size)),
            disable=disable_tqdm,
        ):
            inps = []
            ctxlens = []
            for cache_key, context_enc, continuation_enc in chunk:
                inp = (context_enc + continuation_enc)[-self.max_length :]

                ctxlen = len(context_enc) - max(
                    0, len(context_enc) + len(continuation_enc) - (self.max_length + 1)
                )

                if len(context_enc) + len(continuation_enc) > len(inp):
                    print(
                        f"WARNING: Length of concatenated context ...{repr(cache_key[0][-20:])} and continuation {repr(cache_key[1])} exceeds max length {self.max_length + 1}"
                    )

                inps.append(inp)
                ctxlens.append(ctxlen)

            response = self.megatron_completion(
                model_name=self.model_name,
                prompts=inps,
                max_tokens=0,
                temperature=0.0,
                logprobs=10,
            )

            for (
                logprobs,
                is_max_logprobs,
                ctxlen,
                (cache_key, context_enc, continuation_enc),
            ) in zip(response["logprobs"], response["is_max_logprobs"], ctxlens, chunk):
                answer = get_result(logprobs, is_max_logprobs, ctxlen)

                res.append(answer)

                # partial caching
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)

        return re_ord.get_original(res)

    def greedy_until(self, requests):
        if not requests:
            return []
        res = []

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

            until = request_args["until"]

            for context, _ in chunk:
                context_enc = self.tok_encode(context)
                inp = context_enc[-(self.max_length - self.max_gen_toks) :]
                inps.append(inp)

                if len(context_enc) > len(inp):
                    print(
                        f"WARNING: Length of context ...{repr(context[-20:])} exceeds max length {self.max_length - self.max_gen_toks}"
                    )

            if isinstance(until, str):
                until = [until]

            if until:
                primary_until_tokens = self.tok_encode(until[0], is_continuation=True)
                if len(primary_until_tokens) == 1:
                    primary_until = primary_until_tokens[0]
                else:
                    primary_until = None
            else:
                primary_until = None

            response = self.megatron_completion(
                model_name=self.model_name,
                prompts=inps,
                max_tokens=self.max_gen_toks,
                temperature=0.0,
                stop_token=primary_until,
            )

            until_escaped = [re.escape(u) for u in until]
            until_escaped_or = "|".join(until_escaped)
            until_regex_str = f"(?:{until_escaped_or}).*$"
            until_regex = re.compile(until_regex_str, re.DOTALL)

            for tokens, (context, _), inp in zip(response["tokens"], chunk, inps):
                continuation_enc = tokens[len(inp) :]
                text, continuation = self.tok_decode_batch((tokens, continuation_enc))

                # Check if the result of tokenizing and detokenizing the context differs from the original context
                # to detect problematic samples from the original run
                if not text.startswith(context):
                    print(
                        f"INFO: Decoded text {repr(text[:10])}... does not begin with context {repr(context[:10])}..."
                    )

                s = until_regex.sub("", continuation)

                self.cache_hook.add_partial("greedy_until", (context, until), s)

                res.append(s)

        return re_ord.get_original(res)

    def _model_call(self, inps):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override greedy_until
        raise NotImplementedError()

    def megatron_metadata(self):
        headers = {
            "Content-Type": "application/json",
        }
        response = requests.put(f"{self.server_url}/metadata", headers=headers)

        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.json()['message']}")

        return response.json()

    def tokenizer_query(self, prompts, is_continuation):
        headers = {
            "Content-Type": "application/json",
        }

        data = {
            "prompts": prompts,
            "is_continuation": is_continuation,
        }

        response = requests.put(
            f"{self.server_url}/tokenize", data=json.dumps(data), headers=headers
        )

        # if response.status_code != 200:
        #     print(f"Error {response.status_code}: {response.json()['message']}")
        # else:
        #     print("Megatron Response: ")
        #     print(response.json())

        return response.json()["tokens"]

    def detokenizer_query(self, seqs):
        headers = {
            "Content-Type": "application/json",
        }

        data = {
            "seqs": seqs,
        }

        response = requests.put(
            f"{self.server_url}/detokenize", data=json.dumps(data), headers=headers
        )

        if response.status_code != 200:
            raise Exception(
                f"Error {response.status_code}: {response.json()['message']}"
            )

        response_json = response.json()

        if not isinstance(response_json, dict):
            raise Exception(
                f"Unexpected response {response.status_code}: {response_json}"
            )

        # print("Megatron Response: ")
        # print(response_json)

        return response.json()["text"]

    def megatron_query(
        self,
        model_name,
        prompts,
        max_tokens,
        temperature,
        logprobs=0,
        top_k=None,
        stop_token=None,
    ):
        headers = {
            "Content-Type": "application/json",
        }

        data = {
            "model_name": model_name,
            "prompts": prompts,
            "tokens_to_generate": max_tokens,
            "logprobs": logprobs > 0,
            "stop_token": stop_token if stop_token is not None else self.eot_token_id,
        }

        if temperature == 0:
            # Choose only the most likely token during generation, i.e. greedy search (no sampling)
            data["top_k"] = 1
        else:
            data["temperature"] = temperature
            if top_k is not None:
                data["top_k"] = top_k

        response = requests.put(
            f"{self.server_url}/api", data=json.dumps(data), headers=headers
        )

        if response.status_code != 200:
            raise Exception(f"{response.status_code} {response.content}")

        response_json = response.json()

        if not isinstance(response_json, dict):
            raise Exception(
                f"Unexpected HTTP response {response.status_code}: {repr(response_json)}"
            )

        return response.json()

    def megatron_completion(self, **kwargs):
        """Query Megatron-LM Server API for completion.

        Retry with back-off until they respond
        """

        backoff_time = 3
        while True:
            try:
                return self.megatron_query(**kwargs)
            except Exception:
                import traceback

                traceback.print_exc()
                time.sleep(backoff_time)
                backoff_time *= 1.5
