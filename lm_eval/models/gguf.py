import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from requests.exceptions import RequestException
from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


logger = logging.getLogger(__name__)

# logit bias used to force the server to sample a specific token. llama.cpp
# still reports the token's *pre-sampling* logprob and an unbiased
# top_logprobs list, so this yields exact teacher-forced loglikelihoods.
_FORCE_TOKEN_BIAS = 100


@register_model("gguf", "ggml")
class GGUFLM(LM):
    """Evaluate GGUF models served by a llama.cpp server (llama-server)
    through its OpenAI-compatible endpoints.

    Requires a llama.cpp version that returns logprobs in the modern
    OpenAI format (`logprobs.content`, see llama.cpp PR #10783; any release
    from December 2024 onward). The deprecated legacy format
    (`token_logprobs`/`text_offset`) is not supported.

    Pass `base_url` pointing at the server, e.g.
    `--model gguf --model_args base_url=http://127.0.0.1:8080`.
    When the server runs in router mode (multiple models), also pass
    `model=<name-or-alias>` so requests are routed correctly.

    Requests are issued concurrently with a thread pool. By default the
    degree of parallelism is auto-detected from the server's `/props`
    endpoint (`total_slots`, i.e. llama-server's `--parallel` setting);
    override it with `parallel=<N>`. Note that llama-server only processes
    as many requests simultaneously as it has slots, so values above the
    server's slot count mostly add queueing on the server side.
    """

    def __init__(
        self,
        base_url=None,
        model=None,
        max_length=2048,
        timeout=300,
        temperature=0.0,
        parallel=None,
        **kwargs,
    ):
        super().__init__()
        assert base_url, "must pass `base_url` to use GGUF LM!"
        base_url = base_url.rstrip("/")
        # derive the server root so /props and /tokenize can be queried
        server_url = base_url
        for suffix in ("/v1/completions", "/completions", "/v1"):
            if server_url.endswith(suffix):
                server_url = server_url[: -len(suffix)]
                break
        self.server_url = server_url
        self.completions_url = server_url + "/v1/completions"
        self.tokenize_url = server_url + "/tokenize"
        self.model = model
        self.temperature = temperature
        self.max_length = max_length
        self.timeout = timeout
        self.parallel = parallel
        self._resolved_parallel = None
        self._tok_cache = {}
        self._tok_lock = threading.Lock()

    def _post_with_retries(self, url, payload, retries=3, delay=5):
        for _ in range(retries):
            try:
                response = requests.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except RequestException as e:
                logger.error(f"RequestException: {e}")
                time.sleep(delay)  # wait before retrying
        raise RuntimeError(f"Failed to get a valid response after {retries} retries.")

    def _tokenize(self, text, add_special):
        """Tokenize text with the server's own tokenizer (cached)."""
        key = (text, add_special)
        with self._tok_lock:
            cached = self._tok_cache.get(key)
        if cached is not None:
            return cached
        payload = {"content": text, "add_special": add_special}
        if self.model is not None:
            payload["model"] = self.model
        result = self._post_with_retries(self.tokenize_url, payload)
        ids = result["tokens"]
        with self._tok_lock:
            self._tok_cache[key] = ids
        return ids

    def _detect_total_slots(self):
        """Query the server's /props endpoint for its slot count
        (llama-server's `--parallel` setting). Returns None if unavailable.
        """
        try:
            params = {"model": self.model} if self.model is not None else None
            response = requests.get(
                f"{self.server_url}/props", params=params, timeout=10
            )
            response.raise_for_status()
            total_slots = response.json().get("total_slots")
            if isinstance(total_slots, int) and total_slots > 0:
                return total_slots
        except (RequestException, ValueError) as e:
            logger.debug(f"Could not query /props for slot count: {e}")
        return None

    def _resolve_parallel(self):
        if self._resolved_parallel is None:
            if self.parallel is not None:
                self._resolved_parallel = max(1, int(self.parallel))
            else:
                total_slots = self._detect_total_slots()
                self._resolved_parallel = total_slots or 1
                if total_slots:
                    logger.info(
                        f"Auto-detected {total_slots} llama.cpp server slots; "
                        f"issuing up to {total_slots} concurrent requests. "
                        "Override with `parallel=<N>`."
                    )
        return self._resolved_parallel

    def _map_requests(self, fn, items, disable_tqdm):
        """Apply fn to each item, preserving order, using a thread pool when
        parallelism is enabled.
        """
        parallel = self._resolve_parallel()
        if parallel <= 1 or len(items) <= 1:
            return [fn(item) for item in tqdm(items, disable=disable_tqdm)]
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            return list(
                tqdm(
                    executor.map(fn, items),
                    total=len(items),
                    disable=disable_tqdm,
                )
            )

    def gguf_completion(
        self,
        context,
        continuation=None,
        stop=None,
        max_tokens=None,
        id_slot=None,
        **kwargs,
    ):
        request = {
            "prompt": context,
            "temperature": self.temperature,
        }
        if self.model is not None:
            request["model"] = self.model
        if id_slot is not None:
            # pin the request to a specific server slot so that prompts
            # sharing a prefix hit the same per-slot KV cache
            request["id_slot"] = id_slot
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        if stop is not None:
            request["stop"] = stop
        return self._post_with_retries(self.completions_url, request)

    def _score_position(self, prompt_ids, target_id, id_slot=None):
        """Score one continuation token given its natural preceding context.

        Sends the preceding tokens as a token-id array (exact control over
        tokenization, matching the HF backend's encode-context-and-
        continuation-separately convention) and forces the server to sample
        `target_id` with a large logit bias. llama.cpp reports the sampled
        token's *pre-sampling* logprob and an unbiased top_logprobs list,
        so the returned values are the true teacher-forced logprob of the
        target token and whether it was the argmax.
        """
        request = {
            "prompt": prompt_ids,
            "temperature": 0,
            "max_tokens": 1,
            "logprobs": 2,
            "logit_bias": [[target_id, _FORCE_TOKEN_BIAS]],
        }
        if self.model is not None:
            request["model"] = self.model
        if id_slot is not None:
            request["id_slot"] = id_slot
        response = self._post_with_retries(self.completions_url, request)
        if not (response and response.get("choices")):
            raise RuntimeError(f"Invalid scoring response: {response}")
        content = response["choices"][0].get("logprobs", {}).get("content") or []
        if not content:
            raise RuntimeError(f"Missing logprobs in scoring response: {response}")
        entry = content[0]
        if entry.get("id") != target_id:
            raise RuntimeError(
                f"Server did not sample the forced token id {target_id}: {entry}"
            )
        top_logprobs = entry.get("top_logprobs") or []
        is_greedy = bool(top_logprobs) and (
            max(top_logprobs, key=lambda t: t["logprob"]).get("id") == target_id
        )
        return entry["logprob"], is_greedy

    def _loglikelihood_one(self, item):
        args, id_slot = item
        context, continuation = args
        # tokenize the same way the HF backend does: context and continuation
        # encoded separately (continuation without special tokens)
        continuation_ids = self._tokenize(continuation, add_special=False)
        if not continuation_ids:
            return (0.0, True)
        context_ids = self._tokenize(context, add_special=True)
        total = 0.0
        is_greedy = True
        for j, target_id in enumerate(continuation_ids):
            logprob, greedy = self._score_position(
                context_ids + continuation_ids[:j], target_id, id_slot=id_slot
            )
            total += logprob
            is_greedy = is_greedy and greedy
        return (total, is_greedy)

    @staticmethod
    def _assign_slots_by_context(args_list, parallel):
        """Assign a server slot id to each (context, continuation) pair.

        Consecutive requests sharing the same context (e.g. the candidate
        continuations of one multiple-choice question) are pinned to the same
        slot, so all but the first reuse the slot's cached prompt prefix.
        Groups are round-robined across slots to keep them busy in parallel.
        """
        slots = []
        group_idx = -1
        prev_context = None
        for context, _ in args_list:
            if context != prev_context:
                group_idx += 1
                prev_context = context
            slots.append(group_idx % parallel)
        return slots

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        if not requests:
            return []
        parallel = self._resolve_parallel()
        args_list = [req.args for req in requests]
        slots = self._assign_slots_by_context(args_list, parallel)
        return self._map_requests(
            self._loglikelihood_one,
            list(zip(args_list, slots, strict=True)),
            disable_tqdm,
        )

    def _generate_one(self, args):
        inp, request_args = args
        until = request_args.get("until", ["</s>"])
        max_gen_toks = request_args.get("max_gen_toks", None)
        # no id_slot pinning here: generation lengths vary widely, so the
        # server's dynamic idle-slot assignment load-balances better than a
        # static assignment (measured ~30% slower with pinning on gsm8k)
        response = self.gguf_completion(
            context=inp, stop=until, max_tokens=max_gen_toks
        )
        if response and "choices" in response and response["choices"]:
            choice = response["choices"][0]
            if "text" in choice:
                return choice["text"].strip()
            else:
                logger.error(f"Invalid response for greedy_until. Response: {response}")
                return None  # Add default value in case of error
        else:
            logger.error(f"Invalid response for greedy_until. Response: {response}")
            return None  # Add default value in case of error

    def generate_until(self, requests, disable_tqdm: bool = False):
        if not requests:
            return []

        return self._map_requests(
            self._generate_one,
            [req.args for req in requests],
            disable_tqdm,
        )

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError(
            "loglikelihood_rolling not yet supported for GGUF models"
        )
