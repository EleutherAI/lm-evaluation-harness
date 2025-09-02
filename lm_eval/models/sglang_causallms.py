import copy
import logging
from importlib.util import find_spec
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from tqdm import tqdm

from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import (
    Collator,
    handle_stop_sequences,
    postprocess_generated_text,
)
from lm_eval.utils import (
    get_rolling_token_windows,
    make_disjoint_window,
)


eval_logger = logging.getLogger(__name__)

try:
    import sglang as sgl
except ModuleNotFoundError:
    pass

if TYPE_CHECKING:
    pass


@register_model("sglang")
class SGLangLM(TemplateLM):
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        pretrained: str,
        # batch args from lm-eval interface:  https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md
        batch_size: Union[str, int] = 1,
        max_batch_size=None,
        max_model_len: int = None,
        max_gen_toks: int = 256,
        add_bos_token: Optional[bool] = False,
        ########## SGlang native args ##########
        # Todo(Jinwei): Include more args of SGLang Engine if needed. Refer to https://docs.sglang.ai/backend/server_arguments.html .
        tokenizer_path: Optional[str] = None,
        tokenizer_mode: str = "auto",
        load_format: str = "auto",
        trust_remote_code: bool = True,
        dtype: str = "auto",
        kv_cache_dtype: str = "auto",
        context_length: Optional[int] = None,
        device: str = "cuda",
        chunked_prefill_size: int = -1,
        # Memory and scheduling
        mem_fraction_static: Optional[float] = None,
        # parallelism
        dp_size: int = 1,
        tp_size: int = 1,
        prefix_token_id: Optional[int] = None,
        # End marker for thinking tags - splits to get response after this token (if provided).
        think_end_token: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        if not find_spec("sglang"):
            raise ModuleNotFoundError(
                "attempted to use 'sglang' LM type, but package `sglang` is not installed. "
                "Please install sglang via official document here:https://docs.sglang.ai/start/install.html#install-sglang"
            )

        assert "cuda" in device or device is None, "SGLang only supports CUDA"
        assert context_length is None or max_model_len is None, (
            "Either context_length or max_model_len may be provided, but not both"
        )
        # Initialize your sglang model here
        self.think_end_token = think_end_token
        self._max_length = (
            max_model_len if max_model_len is not None else context_length
        )
        self.tensor_parallel_size = int(tp_size)
        self.data_parallel_size = int(dp_size)
        self.model_args = {
            "model_path": pretrained,
            "tokenizer_path": tokenizer_path,
            "tokenizer_mode": tokenizer_mode,
            "load_format": load_format,
            "trust_remote_code": trust_remote_code,
            "dtype": dtype,
            "kv_cache_dtype": kv_cache_dtype,
            "device": device,
            "mem_fraction_static": mem_fraction_static,
            "tp_size": self.tensor_parallel_size,
            "dp_size": self.data_parallel_size,
            "chunked_prefill_size": chunked_prefill_size,
        }

        self.model_args.update(kwargs)
        self.batch_size = (
            "auto"
            if isinstance(batch_size, str) and "auto" in batch_size
            else int(batch_size)
        )
        if self.data_parallel_size > 1:
            eval_logger.warning(
                "Data parallelism will be deprecated in the future version of SGLang. See here: https://docs.sglang.ai/backend/server_arguments.html#data-parallelism ."
            )
        self.model = sgl.Engine(**self.model_args)

        # Todo(Jinwei): check tokenizer and other settings.
        self.tokenizer = self.model.tokenizer_manager.tokenizer
        self._max_gen_toks = max_gen_toks
        self.add_bos_token = add_bos_token
        if "gemma" in pretrained.lower():
            self.add_bos_token = True
            eval_logger.info(
                "Found 'gemma' in model name, a BOS token will be used as Gemma series models underperform without it."
            )
        self.custom_prefix_token_id = prefix_token_id

    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
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
            rolling_token_windows: List[Tuple[List[int], List[int]]] = list(
                map(
                    make_disjoint_window,
                    get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.prefix_token_id,
                        # max_seq_len - (1 for context)
                        max_seq_len=self.max_length - 1,
                        context_len=1,
                    ),
                )
            )

            # TODO: Right now, we pass single EOT token to the Encoder and the full context to the decoder, in seq2seq case
            windows = [(None,) + x for x in rolling_token_windows]

            # Store windows with their request index
            all_windows.extend((req_idx, window) for window in windows)
            request_window_counts.append(len(windows))

        all_nlls = []
        batch_size = adaptive_batch_size or int(self.batch_size)
        for i in range(0, len(all_windows), batch_size):
            batch = all_windows[i : i + batch_size]
            # Extract just the windows for processing, keeping track of request indices
            batch_indices, batch_windows = zip(*batch)

            batch_nlls = self._loglikelihood_tokens(
                requests=batch_windows,
                disable_tqdm=False,
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

            string = requests[len(loglikelihoods) - 1].args[0]
            self.cache_hook.add_partial(
                "loglikelihood_rolling", (string,), request_total
            )

        return loglikelihoods

    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        res = []

        # batch tokenize contexts
        context, all_gen_kwargs = zip(*(req.args for req in requests))
        context_encoding: List[List[int]] = self.tok_encode(
            context, add_special_tokens=self.add_bos_token
        )
        requests = [
            ((a, b), c) for a, b, c in zip(context, context_encoding, all_gen_kwargs)
        ]

        def _collate_gen(_requests):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            return -len(_requests[0][1]), _requests[0][0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = Collator(requests, _collate_gen, group_by=None)
        chunks = re_ords.get_batched(
            n=int(self.batch_size) if self.batch_size != "auto" else 0, batch_fn=None
        )

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )
        # for each different set of kwargs, we execute all requests, by batch.
        eos = self.tokenizer.decode(self.eot_token_id)
        for chunk in chunks:
            context_and_encoding, all_gen_kwargs = zip(*chunk)
            context, context_encoding = zip(*context_and_encoding)

            context_encoding_truncated = []
            sampling_params = []
            for x, gen_kwargs in zip(context_encoding, all_gen_kwargs):
                # unpack our keyword arguments.
                if isinstance(gen_kwargs, dict):
                    kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                    # add EOS token to stop sequences
                    until = handle_stop_sequences(kwargs.pop("until", None), eos=eos)
                else:
                    raise ValueError(
                        f"Expected `kwargs` to be of type `dict` but got {type(gen_kwargs)}"
                    )
                if "max_gen_toks" in kwargs.keys():
                    max_gen_toks = kwargs.pop("max_gen_toks")
                else:
                    max_gen_toks = self.max_gen_toks

                # set the max length in tokens of inputs ("context_enc")
                # max len for inputs = max length, minus room to generate the max new tokens
                max_ctx_len = self.max_length - max_gen_toks
                if len(x) > max_ctx_len:
                    context_encoding_truncated.append(x[-max_ctx_len:])
                else:
                    context_encoding_truncated.append(x)
                # create sampling params
                kwargs = self.modify_gen_kwargs(kwargs)
                sampling_params.append(
                    kwargs | {"max_tokens": max_gen_toks, "stop": until}
                )
            # perform batched generation
            # cont is a list of dic. See here https://github.com/sgl-project/sglang/blob/0a6f18f068e4095fc228e798454e8496c9749214/python/sglang/srt/entrypoints/engine.py#L111 .
            cont = self._model_generate(
                requests=context_encoding_truncated,
                generate=True,
                sampling_params=sampling_params,
            )

            # cache generations
            for output, context in zip(cont, context):
                generated_text = output.get("text", "")
                generated_text = postprocess_generated_text(
                    generated_text, until, self.think_end_token
                )
                res.append(generated_text)
                self.cache_hook.add_partial(
                    "generate_until", (context, gen_kwargs), generated_text
                )
                pbar.update(1)

        pbar.close()
        # reorder all group of results back to original unsorted form
        return re_ords.get_original(res)

    def _model_generate(
        self,
        requests: List[List[int]] = None,
        generate: bool = False,
        sampling_params: Union[List[Dict], Dict, None] = None,
        return_logprob: bool = False,
        top_logprobs_num: int = 1,
        logprob_start_len: int = -1,
    ):
        # check sglang sampling parameters: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/sampling/sampling_params.py#L21  and https://docs.sglang.ai/references/sampling_params.html.
        if not generate:
            sampling_params = sampling_params if sampling_params else {}
            sampling_params.update(
                {
                    "temperature": 0,
                    "max_new_tokens": 1,
                }
            )
        if not isinstance(sampling_params, List):
            sampling_params = [sampling_params] * len(requests)
        # Refer to:  https://docs.sglang.ai/backend/offline_engine_api.html
        outputs = self.model.generate(
            input_ids=requests,
            sampling_params=sampling_params,
            return_logprob=return_logprob,
            top_logprobs_num=top_logprobs_num,
            logprob_start_len=logprob_start_len,
        )
        return outputs

    @property
    def eot_token_id(self):
        # Return the EOT (End of Text) token ID
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self):
        # it is used as prefix for loglikelihood
        if self.custom_prefix_token_id is not None:
            return self.custom_prefix_token_id
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        if hasattr(self.model, "tokenizer_manager") and hasattr(
            self.model.tokenizer_manager, "context_len"
        ):
            return self.model.tokenizer_manager.context_len
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self):
        # Return the maximum number of tokens for generation
        return self._max_gen_toks

    def tok_encode(
        self,
        string: Union[str, List[str]],
        left_truncate_len: int = None,
        add_special_tokens: bool = False,
        truncation: bool = False,
    ) -> Union[List[int], List[List[int]]]:
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

    def tok_decode(self, tokens: List[int]) -> str:
        # Implement token-to-text decoding
        pass

    @property
    def tokenizer_name(self) -> str:
        """
        Return the name of the model's tokenizer and/or the accompanying chat template.
        The returned string is used to cache requests.

        Returns:
            str: The name of the model's tokenizer and/or chat template.
        """
        pass

    def chat_template(self, chat_template: Union[bool, str] = False) -> str:
        """
        Get the appropriate chat template for the model based on the `chat_template` argument.

        This method returns the chat template string to build the prompt from a chat history.
        The chat template is saved in the evaluation results for reproducibility.
        Boolean arguments should be used with models that have only one chat template,
        while string arguments are used with models that have multiple chat templates.
        For the reference implementation, see HFLM class in `lm_eval.models.huggingface`.

        Args:
            chat_template (Union[bool, str]): Specifies whether to apply a chat template:
                - If False: Do not apply any chat template.
                - If True: Apply the default chat template.
                - If str: Apply the specified chat template by name.

        Returns:
            str: The selected chat template in Jinja format.
        """
        pass

    def apply_chat_template(
        self, chat_history: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        chat_templated = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
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

            outputs = self._model_generate(
                requests=inputs,
                generate=False,
                return_logprob=True,
                top_logprobs_num=2,
                logprob_start_len=0,
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
        :param outputs:
            Contains input_token_logprobs and input_top_logprobs
        :param ctxlen: int
            Length of context (so we can slice them away and only keep the predictions)
        :return:
            continuation_logprobs: float
                Log probabilities of continuation tokens
            is_greedy: bool
                Whether argmax matches given continuation exactly
        """

        # The first entry of prompt_logprobs is None because the model has no previous tokens to condition on.
        # [(logprob, token_id, token_text)]
        continuation_logprobs_lists = outputs["meta_info"]["input_token_logprobs"]
        continuation_logprobs = sum(
            logprob for logprob, _, _ in continuation_logprobs_lists[ctxlen:]
        )

        top_logprobs_lists = outputs["meta_info"]["input_top_logprobs"]

        # Determine if is_greedy
        is_greedy = True
        for token, top_logprobs in zip(tokens[ctxlen:], top_logprobs_lists[ctxlen:]):
            if top_logprobs:
                top_token = max(top_logprobs, key=lambda x: x[0])[1]
                if top_token != token:
                    is_greedy = False
                    break
        return continuation_logprobs, is_greedy

    @staticmethod
    def modify_gen_kwargs(kwargs: dict) -> dict:
        # sampling_params
        kwargs["temperature"] = kwargs.get("temperature", 0.0)
        do_sample = kwargs.pop("do_sample", None)
        if do_sample is False and "temperature" not in kwargs:
            eval_logger.debug(
                "Got `do_sample=False` and no temperature value, setting VLLM temperature to 0.0 ..."
            )
            kwargs["temperature"] = 0.0
        # hf defaults
        kwargs["skip_special_tokens"] = kwargs.get("skip_special_tokens", False)
        kwargs["spaces_between_special_tokens"] = kwargs.get(
            "spaces_between_special_tokens", False
        )
        return kwargs
