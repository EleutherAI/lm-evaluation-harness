"""TensorRT-LLM model wrapper for lm-evaluation-harness."""

import logging
from importlib.util import find_spec
from typing import List, Literal, Optional, Tuple, Union, Any

from tqdm import tqdm

from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import (
    Collator,
    _add_special_kwargs,
    configure_pad_token,
    handle_stop_sequences,
    has_bos_prefix,
    maybe_truncate,
    normalize_gen_kwargs,
    postprocess_generated_text,
)
from lm_eval.utils import (
    get_rolling_token_windows,
    make_disjoint_window,
)
from lm_eval.api.instance import Instance

try:
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig
except ModuleNotFoundError:
    pass

eval_logger = logging.getLogger(__name__)


@register_model("trtllm")
class TRTLLM(TemplateLM):
    """TensorRT-LLM model wrapper for lm-evaluation-harness.

    This class provides an interface to TensorRT-LLM models, inheriting from TemplateLM
    to leverage built-in tokenization and chat template support.
    """

    def __init__(
        self,
        model: str,
        dtype: Literal["float16", "bfloat16", "float32", "auto"] = "auto",
        trust_remote_code: bool = False,
        tokenizer: Optional[str] = None,
        tokenizer_mode: Literal["auto", "slow"] = "auto",
        add_bos_token: Optional[bool] = None,
        prefix_token_id: int | None = None,
        max_batch_size: int = 32,
        max_input_len: int = 2048,
        max_output_len: int = 512,
        max_beam_width: int = 1,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        batch_size: Union[str, int] = "auto",
        max_gen_toks: int = 256,
        seed: int = 1234,
        truncation_side: Literal["left", "right", "middle"] = "left",
        chat_template_args: Optional[dict] = None,
        # TRT-LLM: enable thinking tags in apply_chat_template.
        enable_thinking: bool = True,
        # End marker for thinking tags - splits to get response after this token (if provided).
        think_end_token: str | None = None,
        **kwargs,
    ):
        super().__init__()

        # Check TensorRT-LLM installation
        if not find_spec("tensorrt_llm"):
            raise ModuleNotFoundError(
                "attempted to use 'trtllm' LM type, but package `tensorrt_llm` is not installed. "
                "Please install TensorRT-LLM following NVIDIA's installation guide: "
                "https://github.com/NVIDIA/TensorRT-LLM"
            )

        # Set batch size and max generation tokens
        self.batch_size = (
            "auto"
            if isinstance(batch_size, str) and "auto" in batch_size
            else int(batch_size)
        )
        self._max_gen_toks = max_gen_toks

        # Store configuration
        self.model = model
        self.trust_remote_code = trust_remote_code
        self.max_batch_size = max_batch_size
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.truncation_side = truncation_side
        self.chat_template_args = chat_template_args or {}
        self.enable_thinking = self.chat_template_args.pop(
            "enable_thinking", enable_thinking
        )
        self.think_end_token = think_end_token
        self.custom_prefix_token_id = prefix_token_id

        # Load tokenizer from HuggingFace
        eval_logger.info(f"Loading tokenizer from {tokenizer or model}")
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer if tokenizer else model,
            trust_remote_code=trust_remote_code,
            use_fast=(tokenizer_mode == "auto"),
        )

        # Configure pad token
        self.tokenizer = configure_pad_token(self.tokenizer)

        # Handle BOS token
        self.add_bos_token = add_bos_token
        if add_bos_token is None:
            # Auto-detect BOS token behavior
            try:
                test_tok = self.tok_encode("test", add_special_tokens=True)
                test_tok_no_special = self.tok_encode("test", add_special_tokens=False)
                self.add_bos_token = test_tok != test_tok_no_special
            except Exception:
                self.add_bos_token = False

        eval_logger.info(f"BOS token will{' not' if not self.add_bos_token else ''} be added")

        if prefix_token_id is not None:
            eval_logger.info(
                f"Loglikelihood prefix token id used in evaluation: {self.prefix_token_id}"
            )

        # Initialize TRT-LLM model using high-level API
        # The LLM class handles engine building and caching automatically
        eval_logger.info(f"Initializing TensorRT-LLM for {model}...")

        # If reuse kv cache, no logprobs can be computed.
        kv_cache_config = KvCacheConfig(
            enable_block_reuse=False,
        )
        # Build kwargs for LLM constructor
        llm_kwargs = {
            "model": model,
            "tensor_parallel_size": tensor_parallel_size,
            "pipeline_parallel_size": pipeline_parallel_size,
            "trust_remote_code": trust_remote_code,
            "kv_cache_config": kv_cache_config,
            "max_input_len": max_input_len,
            "max_seq_len": max_input_len + max_output_len,
            "max_batch_size": max_batch_size,
        }
        if dtype != "auto":
            llm_kwargs["dtype"] = dtype
        self.llm = LLM(**llm_kwargs)
        eval_logger.info(f"TensorRT-LLM model loaded: {model} ")

    def tok_encode(
        self,
        string: str | list[str],
        add_special_tokens: bool | None = None,
        **kwargs,
    ) -> list[int] | list[list[int]]:
        assert self.tokenizer
        if not string:
            return []

        _string: list[str] = [string] if isinstance(string, str) else list(string)
        _bos_token = self.tokenizer.decode(self.prefix_token_id)

        special_tokens_kwargs = {
            **kwargs,
            **_add_special_kwargs(add_special_tokens, self.add_bos_token),
        }

        # Handle chat templates that may already include BOS token.
        # Split strings into those with/without BOS prefix to avoid double-BOS.
        has_prefix_flags = [has_bos_prefix(s, _bos_token) for s in _string]
        idx_has = [i for i, f in enumerate(has_prefix_flags) if f]
        idx_not = [i for i, f in enumerate(has_prefix_flags) if not f]

        strs_has = [_string[i] for i in idx_has]
        strs_not = [_string[i] for i in idx_not]

        enc_has = []
        # If the text already has BOS, do not add special tokens (to avoid double BOS).
        if strs_has:
            kwargs_off = {**special_tokens_kwargs, "add_special_tokens": False}
            enc_has = self.tokenizer(
                strs_has, return_attention_mask=False, **kwargs_off
            ).input_ids

        enc_not = (
            self.tokenizer(
                strs_not, return_attention_mask=False, **special_tokens_kwargs
            ).input_ids
            if strs_not
            else []
        )

        out: list[list[int]] = [None] * len(_string)  # type: ignore
        for j, i in enumerate(idx_has):
            out[i] = enc_has[j]
        for j, i in enumerate(idx_not):
            out[i] = enc_not[j]

        return out[0] if isinstance(string, str) else out

    def apply_chat_template(
        self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        import jinja2

        try:
            chat_templated = self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
                enable_thinking=self.enable_thinking,
                **self.chat_template_args,
            )
        except jinja2.exceptions.TemplateError:
            eval_logger.warning(
                "Failed to apply chat template. Removing the system role in chat history."
            )
            chat_templated = self.tokenizer.apply_chat_template(
                [msg for msg in chat_history if msg["role"] != "system"],
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
                enable_thinking=self.enable_thinking,
                **self.chat_template_args,
            )

        return chat_templated

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
    ) -> List[Tuple[float, bool]]:
        res = []

        def _collate(x):
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        # Reorder requests by length and batch
        re_ord = Collator(requests, sort_fn=_collate)
        chunks = re_ord.get_batched(
            n=int(self.batch_size) if self.batch_size != "auto" else 0, batch_fn=None
        )
        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc="Running loglikelihood requests",
        )
        for chunk in chunks:
            inputs = []
            ctxlens = []
            for cache_key, context_enc, continuation_enc in chunk:
                inp = (context_enc + continuation_enc)[-(self.max_length) :]
                ctxlen = len(context_enc) - max(
                    0, len(context_enc) + len(continuation_enc) - (self.max_length)
                )

                inputs.append(inp)
                ctxlens.append(ctxlen)

            outputs = self.llm.generate(
                inputs=inputs,
                sampling_params=SamplingParams(
                    max_tokens=1,
                    temperature=0,
                    prompt_logprobs=1,
                )
            )
            for output, ctxlen, (cache_key, _, _), inp in zip(
                outputs, ctxlens, chunk, inputs
            ):
                answer = self._parse_logprobs(
                    tokens=inp,
                    outputs=output,
                    ctxlen=ctxlen,
                )
                res.append(answer)

                if cache_key is not None:
                    # special case: loglikelihood_rolling produces a number of loglikelihood requests
                    # all with cache key None. instead do add_partial on the per-example level
                    # in the loglikelihood_rolling() function for those.
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)
                pbar.update(1)
        pbar.close()
        return re_ord.get_original(res)

    @staticmethod
    def _parse_logprobs(tokens: List, outputs, ctxlen: int) -> Tuple[float, bool]:
        """Process logprobs and tokens.

        :param tokens: list
            Input tokens (potentially left-truncated)
        :param outputs: RequestOutput
            outputs.outputs[0].prompt_logprobs is TokenLogprobs = list[dict[int, Logprob]]
            - No None entries (unlike vLLM); one dict per prompt token position.
            - With prompt_logprobs=0: each dict has 1 entry (the actual token's logprob).
            - With prompt_logprobs=K>0: top-K tokens + actual token (if not in top-K).
        :param ctxlen: int
            Length of context (so we can slice them away and only keep the predictions)
        :return:
            continuation_logprobs: float
                Log probabilities of continuation tokens
            is_greedy: bool
                Whether argmax matches given continuation exactly
        """
        # We need to shift for one token to get the logprobs of the continuation tokens.
        # TRTLLM returns {current_token_logprob, next_top_token_logprob}
        shifted_prompt_logprobs = [None]
        next_top_token_logprob = None
        for i in range(len(tokens)):
            prompt_logprob = outputs.outputs[0].prompt_logprobs[i]
            current_token_logprob = prompt_logprob[tokens[i]]
            packed_logprob = {tokens[i]: current_token_logprob}
            if next_top_token_logprob is not None:
                packed_logprob.update(next_top_token_logprob)
                shifted_prompt_logprobs.append(packed_logprob)
            prompt_logprob.pop(tokens[i])
            next_top_token_logprob = prompt_logprob

        continuation_logprobs_dicts = shifted_prompt_logprobs[ctxlen:]
        continuation_logprobs = 0.0
        for token, logprob_dict in zip(tokens[ctxlen:], continuation_logprobs_dicts):
            if token in logprob_dict:
                continuation_logprobs += logprob_dict[token].logprob
            else:
                # Should not happen since TRT-LLM always includes the actual token.
                eval_logger.warning(
                    f"Token {token} not found in prompt_logprobs dict at this position. "
                    "Consider increasing prompt_logprobs value."
                )

        # Determine is_greedy: the actual token must have rank=1 at every position.
        # The Logprob object has a .rank attribute (1 = highest probability).
        is_greedy = True
        for token, logprob_dict in zip(tokens[ctxlen:], continuation_logprobs_dicts):
            if token in logprob_dict and logprob_dict[token].rank is not None:
                if logprob_dict[token].rank != 1:
                    is_greedy = False
                    break
            elif logprob_dict:
                # Fallback: check by finding the max logprob token in the dict
                top_token = max(logprob_dict, key=lambda t: logprob_dict[t].logprob)
                if top_token != token:
                    is_greedy = False
                    break

        return continuation_logprobs, is_greedy

    def generate_until(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[str]:
        assert self.tokenizer
        res = []

        # batch tokenize contexts
        context, all_gen_kwargs = zip(*(req.args for req in requests), strict=True)
        context_encoding = self.tok_encode(context)
        reqs = [
            ((a, b), c)
            for a, b, c in zip(context, context_encoding, all_gen_kwargs, strict=True)
        ]

        def _collate_gen(_requests):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            return -len(_requests[0][1]), _requests[0][0]

        re_ords = Collator(
            reqs,
            _collate_gen,
            group_by=None,
        )
        chunks = re_ords.get_batched(
            n=int(self.batch_size) if self.batch_size != "auto" else 0, batch_fn=None
        )

        pbar = tqdm(
            total=len(reqs),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )
        # for each different set of kwargs, we execute all requests, by batch.
        eos = self.tokenizer.decode(self.eot_token_id)
        for chunk in chunks:
            context_and_encoding, all_gen_kwargs = zip(*chunk, strict=True)
            context, context_encoding = zip(*context_and_encoding, strict=True)
            context_encoding_truncated = []
            sampling_params = []
            _cache_gen_kwargs = []
            for toks, gen_kwargs in zip(context_encoding, all_gen_kwargs, strict=True):
                assert isinstance(gen_kwargs, dict), (
                    f"Expected `gen_kwargs` to be of type `dict` but got {type(gen_kwargs)}"
                )

                kwargs, until, max_gen_toks = self.modify_gen_kwargs(
                    gen_kwargs, eos=eos, default_max_gen_toks=self.max_gen_toks
                )

                # set the max length in tokens of inputs ("context_enc")
                # max len for inputs = max length, minus room to generate the max new tokens
                toks, max_gen_toks = maybe_truncate(
                    toks,
                    max_gen_toks=max_gen_toks,
                    max_model_len=self.max_input_len + self.max_output_len,
                    side=self.truncation_side,
                    verbose=True,
                )
                context_encoding_truncated.append(toks)

                sampling_params.append(
                    SamplingParams(max_tokens=max_gen_toks, stop=until, **kwargs)
                )
                _cache_gen_kwargs.append(
                    kwargs | {"until": until, "max_gen_toks": max_gen_toks}
                )

            # perform batched generation
            cont = self.llm.generate(
                inputs=context_encoding_truncated,
                sampling_params=sampling_params,
                use_tqdm=False,
            )

            # cache generations
            for output, _context, _gen_kwargs in zip(
                cont, context, _cache_gen_kwargs, strict=True
            ):
                generated_text: str = output.outputs[0].text
                # use secondary stop seqs to cut off should-have-been-stopped content post-hoc
                generated_text = postprocess_generated_text(
                    generated_text, _gen_kwargs.get("until"), self.think_end_token
                )
                res.append(generated_text)
                self.cache_hook.add_partial(
                    "generate_until", (_context, _gen_kwargs), generated_text
                )
                pbar.update(1)

        pbar.close()
        # reorder all group of results back to original unsorted form
        return re_ords.get_original(res)

    def loglikelihood_rolling(
        self,
        requests,
        disable_tqdm: bool = False,
        override_bs: Optional[int] = None,
    ) -> List[float]:
        """Compute perplexity using rolling windows.

        This method efficiently batches all rolling windows from all requests together
        for better throughput, following the pattern used in vLLM and SGLang implementations.

        Args:
            requests: List of Instance objects with (string,) in args
            disable_tqdm: Disable progress bar
            override_bs: Override batch size

        Returns:
            List of log-likelihoods
        """
        adaptive_batch_size = None
        if self.batch_size == "auto":
            adaptive_batch_size = len(requests)

        # First, collect all windows from all requests
        all_windows = []  # List of (request_idx, window) tuples
        request_window_counts = []  # Track number of windows per request

        for req_idx, (string,) in enumerate(
            tqdm(
                [req.args for req in requests],
                disable=(disable_tqdm or (self.rank != 0)),
            )
        ):
            # Tokenize the string
            token_list = self.tok_encode(string)

            # Create rolling windows using the utility function
            # This handles the windowing logic properly with context
            rolling_token_windows: List[Tuple[List[int], List[int]]] = list(
                map(
                    make_disjoint_window,
                    get_rolling_token_windows(
                        token_list=token_list,
                        prefix_token=self.prefix_token_id,
                        # max_seq_len - (1 for context) - (1 for generation)
                        max_seq_len=self.max_length - 2,
                        context_len=1,
                    ),
                )
            )

            # Convert to the format expected by _loglikelihood_tokens
            # Format: (None, context_tokens, continuation_tokens)
            # None is used as cache_key to indicate these are rolling windows
            windows = [(None,) + window for window in rolling_token_windows]

            # Store windows with their request index
            all_windows.extend((req_idx, window) for window in windows)
            request_window_counts.append(len(windows))

        # Process all windows in batches
        all_nlls = []
        batch_size = adaptive_batch_size or int(self.batch_size)

        for i in range(0, len(all_windows), batch_size):
            batch = all_windows[i : i + batch_size]
            # Extract just the windows for processing, keeping track of request indices
            batch_indices, batch_windows = zip(*batch)

            # Process the batch using _loglikelihood_tokens
            batch_nlls = self._loglikelihood_tokens(
                requests=batch_windows,
                disable_tqdm=True,  # We already have a progress bar above
            )

            # Store results with their request indices
            all_nlls.extend(zip(batch_indices, batch_nlls))

        # Reconstruct per-request loglikelihoods
        loglikelihoods = []
        current_idx = 0
        for window_count in request_window_counts:
            # Get all nlls for this request
            request_nlls = all_nlls[current_idx : current_idx + window_count]
            # Sum up the nlls for this request (discarding is_greedy)
            request_total = sum(nll[0] for _, nll in request_nlls)
            loglikelihoods.append(request_total)
            current_idx += window_count

            # Add to cache
            string = requests[len(loglikelihoods) - 1].args[0]
            self.cache_hook.add_partial(
                "loglikelihood_rolling", (string,), request_total
            )

        return loglikelihoods

    @property
    def eot_token_id(self) -> int:
        """End-of-text token ID."""
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self) -> int:
        # it is used as prefix for loglikelihood
        if self.custom_prefix_token_id is not None:
            return self.custom_prefix_token_id
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        return self.max_input_len

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def tokenizer_name(self) -> str:
        """Tokenizer name for caching."""
        return self.tokenizer.name_or_path.replace("/", "__")

    @property
    def vocab_size(self) -> int:
        """Vocabulary size."""
        return len(self.tokenizer)

    def __del__(self):
        """Cleanup TRT-LLM resources."""
        if hasattr(self, 'llm'):
            del self.llm

    @staticmethod
    def modify_gen_kwargs(
        gen_kwargs: dict[str, Any],
        eos: str | list[str] | None = None,
        default_max_gen_toks: int = 256,
    ) -> tuple[dict[str, Any], list[str], int]:
        """Process generation kwargs into vLLM-compatible format.

        Args:
            gen_kwargs: Raw generation kwargs from the request.
            eos: EOS token string for stop sequence handling.
            default_max_gen_toks: Default max tokens if not specified in gen_kwargs.

        Returns:
            A tuple of (kwargs, stop_sequences, max_gen_toks) where:
            - kwargs: Processed kwargs ready for SamplingParams
            - stop_sequences: List of stop sequences including EOS
            - max_gen_toks: Maximum tokens to generate
        """
        _gen_kwargs = normalize_gen_kwargs(
            gen_kwargs, default_max_gen_toks=default_max_gen_toks
        )

        # Extract and process stop sequences
        until = handle_stop_sequences(
            _gen_kwargs.pop("until", None), eos=eos[0] if isinstance(eos, list) else eos
        )

        # Extract max_tokens
        max_gen_toks = int(_gen_kwargs.pop("max_gen_toks", default_max_gen_toks))

        # do_sample and temperature normalization is handled by `normalize_gen_kwargs` utility
        _gen_kwargs.pop("do_sample", None)
        # HF defaults
        _gen_kwargs = {
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
        } | _gen_kwargs
        return _gen_kwargs, until, max_gen_toks
