"""TensorRT-LLM model wrapper for lm-evaluation-harness.

This module provides integration with NVIDIA's TensorRT-LLM inference engine,
enabling high-performance evaluation of language models using optimized TensorRT backends.

Supports:
- Loading models from HuggingFace checkpoints with automatic engine building
- Tensor Parallelism for multi-GPU inference
- Loglikelihood scoring and text generation
- Advanced sampling parameters (temperature, top-k, top-p, beam search)
- Engine caching for fast subsequent runs
"""

import hashlib
import logging
import os
from importlib.util import find_spec
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

from tqdm import tqdm

from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import (
    Collator,
    configure_pad_token,
    normalize_gen_kwargs,
)
from lm_eval.utils import (
    get_rolling_token_windows,
    make_disjoint_window,
)

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
        trust_remote_code: bool = False,
        tokenizer: Optional[str] = None,
        tokenizer_mode: Literal["auto", "slow"] = "auto",
        add_bos_token: Optional[bool] = None,
        max_batch_size: int = 32,
        max_input_len: int = 2048,
        max_output_len: int = 512,
        max_beam_width: int = 1,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        batch_size: Union[str, int] = "auto",
        max_gen_toks: int = 256,
        chat_template_args: dict = {},
        **kwargs,
    ):
        """Initialize TensorRT-LLM model wrapper.

        Args:
            model: HuggingFace model identifier or path to local checkpoint
            trust_remote_code: Allow custom code from HuggingFace
            tokenizer: Override tokenizer path (defaults to model)
            tokenizer_mode: Tokenizer loading mode
            add_bos_token: Whether to add BOS token (auto-detected if None)
            dtype: Model precision (float16, bfloat16, float32)
            max_batch_size: Maximum batch size for TRT engine
            max_input_len: Maximum input sequence length
            max_output_len: Maximum output sequence length
            max_beam_width: Maximum beam width for beam search
            tensor_parallel_size: Number of GPUs for tensor parallelism
            pipeline_parallel_size: Number of GPUs for pipeline parallelism
            batch_size: Evaluation batch size
            max_gen_toks: Default maximum generation tokens
            **kwargs: Additional arguments passed to TemplateLM
        """
        super().__init__()

        # Set batch size and max generation tokens
        self.batch_size = batch_size
        self._max_gen_toks = max_gen_toks

        # Check TensorRT-LLM installation
        if not find_spec("tensorrt_llm"):
            raise ModuleNotFoundError(
                "attempted to use 'trtllm' LM type, but package `tensorrt_llm` is not installed. "
                "Please install TensorRT-LLM following NVIDIA's installation guide: "
                "https://github.com/NVIDIA/TensorRT-LLM"
            )

        # Store configuration
        self.model = model
        self.trust_remote_code = trust_remote_code
        self.max_batch_size = max_batch_size
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.chat_template_args = chat_template_args

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
        self._add_bos_token = add_bos_token
        if add_bos_token is None:
            # Auto-detect BOS token behavior
            try:
                test_tok = self.tok_encode("test", add_special_tokens=True)
                test_tok_no_special = self.tok_encode("test", add_special_tokens=False)
                self._add_bos_token = test_tok != test_tok_no_special
            except Exception:
                self._add_bos_token = False

        eval_logger.info(f"BOS token will{' not' if not self._add_bos_token else ''} be added")

        # Initialize TRT-LLM model using high-level API
        # The LLM class handles engine building and caching automatically
        eval_logger.info(f"Initializing TensorRT-LLM for {model}...")

        # If reuse kv cache, no logprobs can be computed.
        kv_cache_config = KvCacheConfig(
            enable_block_reuse=False,
        )
        # Initialize LLM with the model path
        # LLM handles engine building and loading automatically
        # Pass build configuration parameters via kwargs
        self.llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            kv_cache_config=kv_cache_config,
            # Don't pass tokenizer or skip_tokenizer_init - let LLM load its own tokenizer for detokenization
            # Build configuration parameters
            max_input_len=max_input_len,
            max_seq_len=max_input_len + max_output_len,  # Total max sequence length
            max_batch_size=max_batch_size,
        )
        eval_logger.info(f"TensorRT-LLM model loaded: {model} ")

    def tok_encode(
        self,
        string: Union[str, List[str]],
        add_special_tokens: bool = False,
        **kwargs,
    ) -> Union[List[int], List[List[int]]]:
        """Tokenize string(s) to token IDs.

        Args:
            string: Input string or list of strings
            add_special_tokens: Whether to add special tokens (BOS/EOS)
            **kwargs: Additional tokenizer arguments

        Returns:
            Token IDs (list) or batch of token IDs (list of lists)
        """
        # Handle special tokens based on add_bos_token setting
        if add_special_tokens and not self._add_bos_token:
            add_special_tokens = False

        if isinstance(string, str):
            return self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        else:
            return self.tokenizer(
                string,
                add_special_tokens=add_special_tokens,
                **kwargs,
            ).input_ids

    def apply_chat_template(
        self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """Apply chat template to conversation history.

        Args:
            chat_history: List of message dicts with 'role' and 'content' keys
            add_generation_prompt: Whether to add generation prompt at the end

        Returns:
            Formatted string with chat template applied
        """
        import jinja2

        try:
            chat_templated = self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
                **self.chat_template_args,
            )
        except jinja2.exceptions.TemplateError:
            eval_logger.warning(
                "Failed to apply chat template. Removing the system role in chat history."
            )
            chat_history = [msg for msg in chat_history if msg["role"] != "system"]
            chat_templated = self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
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
        self,
        requests,
        disable_tqdm: bool = False,
        override_bs: Optional[int] = None,
    ) -> List[str]:
        """Generate text until stop sequences are encountered.

        Args:
            requests: List of Instance objects with (context, generation_kwargs) in args
            disable_tqdm: Disable progress bar
            override_bs: Override batch size

        Returns:
            List of generated strings
        """
        results = []

        batch_size = override_bs if override_bs is not None else self.batch_size

        # Extract args from Instance objects
        # Each request.args is a tuple of (context, generation_kwargs)
        reqs = [req.args for req in requests]

        # Group by generation kwargs for efficient batching
        collator = Collator(
            reqs,
            sort_fn=lambda x: -len(x[0]),
            group_fn=lambda x: str(x[1]),  # Group by gen_kwargs (convert to string for grouping)
        )

        for chunk in tqdm(
            collator.get_batched(n=batch_size, batch_fn=None),
            disable=disable_tqdm,
            desc="Running generate_until requests",
        ):
            contexts, all_gen_kwargs = zip(*chunk)
            gen_kwargs = all_gen_kwargs[0]  # Same for all in group

            # Normalize generation kwargs
            gen_kwargs = normalize_gen_kwargs(gen_kwargs, self._max_gen_toks)

            # Extract parameters
            max_gen_toks = gen_kwargs.get("max_gen_toks", self._max_gen_toks)
            temperature = gen_kwargs.get("temperature", 0.0)
            top_k = gen_kwargs.get("top_k", -1)
            top_p = gen_kwargs.get("top_p", 1.0)
            repetition_penalty = gen_kwargs.get("repetition_penalty", 1.0)
            num_beams = gen_kwargs.get("num_beams", 1)
            length_penalty = gen_kwargs.get("length_penalty", 1.0)
            until = gen_kwargs.get("until", [])

            # Tokenize contexts
            context_token_ids = []
            for context in contexts:
                tokens = self.tok_encode(context, add_special_tokens=True)

                # Truncate if needed
                if len(tokens) > self.max_input_len:
                    eval_logger.warning(
                        f"Context length {len(tokens)} exceeds max_input_len {self.max_input_len}. "
                        "Truncating from left."
                    )
                    tokens = tokens[-self.max_input_len:]

                context_token_ids.append(tokens)

            # Convert stop sequences to token IDs
            stop_token_ids = None
            if until:
                stop_token_ids = []
                for stop_seq in until:
                    # Tokenize stop sequence
                    tokens = self.tokenizer.encode(stop_seq, add_special_tokens=False)
                    if tokens:
                        stop_token_ids.extend(tokens)
                # Remove duplicates
                stop_token_ids = list(set(stop_token_ids)) if stop_token_ids else None

            sampling_params = SamplingParams(
                end_id=self.eot_token_id,
                pad_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.eot_token_id,
                max_tokens=max_gen_toks,
                temperature=temperature if temperature > 0 else None,
                top_k=top_k if top_k > 0 else None,
                top_p=top_p if top_p < 1.0 else None,
                repetition_penalty=repetition_penalty if repetition_penalty != 1.0 else None,
                use_beam_search=(num_beams > 1),
                length_penalty=length_penalty if length_penalty != 1.0 else None,
                stop_token_ids=stop_token_ids,
            )

            # Generate
            outputs = self.llm.generate(
                inputs=context_token_ids,
                sampling_params=sampling_params,
                use_tqdm=False,
            )

            # Debug logging
            eval_logger.debug(f"Generated {len(outputs)} outputs")

            # Extract generated text
            for idx, output in enumerate(outputs):
                # Get generated text from the first completion output
                # RequestOutput.outputs is a list of CompletionOutput objects
                if hasattr(output, 'outputs') and len(output.outputs) > 0:
                    completion = output.outputs[0]
                    generated_text = completion.text
                    token_ids = completion.token_ids if hasattr(completion, 'token_ids') else []
                    eval_logger.debug(f"Output {idx}: text='{generated_text[:50] if generated_text else '(empty)'}', tokens={len(token_ids)}, finish_reason={getattr(completion, 'finish_reason', None)}")
                else:
                    generated_text = ""
                    eval_logger.debug(f"Output {idx}: No outputs attribute or empty outputs list")

                # Post-process: remove stop sequences if not already removed
                if until:
                    for stop_seq in until:
                        if stop_seq in generated_text:
                            generated_text = generated_text.split(stop_seq)[0]

                results.append(generated_text)
                self.cache_hook.add_partial(
                    "generate_until", (contexts[idx], gen_kwargs), generated_text
                )

        # Reorder results to match original request order
        return collator.get_original(results)

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
    def prefix_token_id(self) -> Optional[int]:
        """Prefix token ID (BOS)."""
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        """Maximum input length."""
        return self.max_input_len

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
