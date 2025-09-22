import copy
import gc
import logging
import os
from importlib.metadata import version
from importlib.util import find_spec
from multiprocessing import Process, Queue
from queue import Empty
from time import sleep
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

import jinja2
from more_itertools import distribute
from packaging.version import parse as parse_version
from tqdm import tqdm

from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import (
    Collator,
    configure_pad_token,
    handle_stop_sequences,
    postprocess_generated_text,
    undistribute,
)
from lm_eval.utils import (
    get_rolling_token_windows,
    make_disjoint_window,
)


try:
    import ray
    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.lora.request import LoRARequest
    from vllm.transformers_utils.tokenizer import get_tokenizer
    from vllm.utils import get_open_port

    if parse_version(version("vllm")) >= parse_version("0.8.3"):
        from vllm.entrypoints.chat_utils import resolve_hf_chat_template
except ModuleNotFoundError:
    pass

if TYPE_CHECKING:
    pass

eval_logger = logging.getLogger(__name__)


def _vllm_mp_worker(
    model_args: dict,
    sampling_params: list["SamplingParams"],
    requests: list[list[int]],
    lora_request: "LoRARequest",
    result_queue: "Queue",
    dp_size: int,
    local_dp_rank: int,
    dp_master_port: int,
    dp_master_ip: str = "127.0.0.1",
) -> None:
    """
    Worker process for vLLM multiprocessing.
    Initializes a vLLM engine, processes requests, and puts results or errors
    onto the result_queue.
    """

    if not requests:
        result_queue.put((local_dp_rank, []))
        return None

    os.environ["VLLM_DP_RANK"] = os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = str(dp_master_ip)
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    llm = None
    try:
        llm = LLM(**model_args)
        res = llm.generate(
            [TokensPrompt(prompt_token_ids=request) for request in requests],
            sampling_params=sampling_params,
            lora_request=lora_request,
        )
        # Give engines time to pause their processing loops before exiting."
        sleep(1)
        result_queue.put((local_dp_rank, res))

    except Exception as e:
        error_message = f"Worker {local_dp_rank} failed during generation: {type(e).__name__}: {str(e)}"
        eval_logger.error(error_message, exc_info=True)
        result_queue.put((local_dp_rank, {"error": error_message}))

    finally:
        if llm is not None:
            try:
                del llm
                gc.collect()
            except Exception as e_cleanup:
                eval_logger.warning(
                    f"Worker {local_dp_rank} encountered an error during LLM cleanup: {type(e_cleanup).__name__}: {str(e_cleanup)}",
                    exc_info=True,
                )

    return None


@register_model("vllm")
class VLLM(TemplateLM):
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        pretrained: str,
        dtype: Literal["float16", "bfloat16", "float32", "auto"] = "auto",
        revision: Optional[str] = None,
        trust_remote_code: Optional[bool] = False,
        tokenizer: Optional[str] = None,
        tokenizer_mode: Literal["auto", "slow"] = "auto",
        tokenizer_revision: Optional[str] = None,
        add_bos_token: Optional[bool] = False,
        prefix_token_id: Optional[int] = None,
        tensor_parallel_size: int = 1,
        quantization: Optional[str] = None,
        max_gen_toks: int = 256,
        swap_space: int = 4,
        batch_size: Union[str, int] = 1,
        max_batch_size=None,
        max_length: int = None,
        max_model_len: int = None,
        seed: int = 1234,
        gpu_memory_utilization: float = 0.9,
        data_parallel_size: int = 1,
        lora_local_path: str = None,
        # VLLM: enable thinking tags in the prompt.
        enable_thinking: bool = True,
        chat_template_args: Optional[dict] = None,
        # End marker for thinking tags - splits to get response after this token (if provided).
        think_end_token: Optional[str] = None,
        max_lora_rank: int = 16,
        **kwargs,
    ):
        super().__init__()

        if not find_spec("vllm"):
            raise ModuleNotFoundError(
                "attempted to use 'vllm' LM type, but package `vllm` is not installed. "
                "Please install vllm via `pip install lm-eval[vllm]` or `pip install -e .[vllm]`"
            )

        assert max_length is None or max_model_len is None, (
            "Either max_length or max_model_len may be provided, but not both"
        )
        kwargs.pop("device", None)
        self.think_end_token = think_end_token
        self.V1 = os.environ.get("VLLM_USE_V1", "1") != "0"
        self._max_length = max_model_len if max_model_len is not None else max_length
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.data_parallel_size = int(data_parallel_size)
        self.model_args = {
            "model": pretrained,
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "revision": revision,
            "dtype": dtype,
            "tokenizer": tokenizer,
            "tokenizer_mode": tokenizer_mode,
            "tokenizer_revision": tokenizer_revision,
            "trust_remote_code": trust_remote_code,
            "tensor_parallel_size": int(tensor_parallel_size),
            "max_model_len": int(self._max_length) if self._max_length else None,
            "max_num_seqs": kwargs.get("max_num_seqs", max_batch_size),
            "swap_space": int(swap_space),
            "quantization": quantization,
            "seed": int(seed),
            "enable_lora": True if lora_local_path else False,
            "max_lora_rank": int(max_lora_rank),
        }
        self.model_args.update(kwargs)
        self.batch_size = (
            "auto"
            if isinstance(batch_size, str) and "auto" in batch_size
            else int(batch_size)
        )
        if self.data_parallel_size <= 1:
            self.model = LLM(**self.model_args)
        else:
            eval_logger.warning(
                "You might experience occasional issues with model weight downloading when data_parallel is in use. To ensure stable performance, run with data_parallel_size=1 until the weights are downloaded and cached."
            )
            self.model_args["distributed_executor_backend"] = (
                "ray"
                if not self.V1
                else self.model_args.get("distributed_executor_backend", None)
            )
            self.batch_size = "auto"
            eval_logger.info("Manual batching is not compatible with data parallelism.")

        if "gemma" in pretrained.lower():
            add_bos_token = True
            eval_logger.info(
                "Found 'gemma' in model name, a BOS token will be used as Gemma series models underperform without it."
            )

        from transformers import AutoConfig

        self._config = AutoConfig.from_pretrained(
            pretrained, trust_remote_code=trust_remote_code, revision=revision
        )
        self.tokenizer = get_tokenizer(
            tokenizer if tokenizer else pretrained,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            revision=tokenizer_revision,
            add_bos_token=add_bos_token,
        )
        self.tokenizer = configure_pad_token(self.tokenizer, model_config=self._config)
        self.chat_template_args = chat_template_args or {}
        self.enable_thinking = self.chat_template_args.pop(
            "enable_thinking", enable_thinking
        )
        self.add_bos_token = add_bos_token

        if parse_version(version("vllm")) >= parse_version("0.8.3"):
            kwargs_resolve_hf_chat_template = {
                "tokenizer": self.tokenizer,
                "chat_template": None,
                "tools": None,
            }

            if parse_version(version("vllm")) >= parse_version("0.9.0"):
                if self.data_parallel_size <= 1:
                    kwargs_resolve_hf_chat_template["model_config"] = (
                        self.model.llm_engine.model_config
                    )
                else:
                    from vllm.engine.arg_utils import EngineArgs

                    engine_args = EngineArgs(**self.model_args)
                    model_config = engine_args.create_model_config()

                    kwargs_resolve_hf_chat_template["model_config"] = model_config
            else:
                kwargs_resolve_hf_chat_template["trust_remote_code"] = trust_remote_code

            self.hf_chat_template = resolve_hf_chat_template(
                **kwargs_resolve_hf_chat_template
            )
        else:
            self.hf_chat_template = None

        self.custom_prefix_token_id = prefix_token_id
        if prefix_token_id is not None:
            eval_logger.info(
                f"Loglikelihood prefix token id used in evaluation: {self.prefix_token_id}"
            )

        self._max_gen_toks = max_gen_toks

        if lora_local_path is not None:
            assert parse_version(version("vllm")) > parse_version("0.3.0"), (
                "lora adapters only compatible with vllm > v0.3.0."
            )
            self.lora_request = LoRARequest("finetuned", 1, lora_local_path)
        else:
            self.lora_request = None

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
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
        if self.data_parallel_size <= 1:
            return self.model.llm_engine.model_config.max_model_len
        else:
            seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
            for attr in seqlen_config_attrs:
                if hasattr(self._config, attr):
                    return getattr(self._config, attr)
            if hasattr(self.tokenizer, "model_max_length"):
                if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                    return self._DEFAULT_MAX_LENGTH
                return self.tokenizer.model_max_length
            return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    def apply_chat_template(
        self, chat_history: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        try:
            chat_templated = self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
                chat_template=self.hf_chat_template,
                enable_thinking=self.enable_thinking,
                **self.chat_template_args,
            )
        except jinja2.exceptions.TemplateError:
            eval_logger.warning(
                "Failed to apply chat template. removing the system role in chat history."
            )
            chat_templated = self.tokenizer.apply_chat_template(
                [msg for msg in chat_history if msg["role"] != "system"],
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
                chat_template=self.hf_chat_template,
                enable_thinking=self.enable_thinking,
                **self.chat_template_args,
            )

        return chat_templated

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

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

    def _model_generate(
        self,
        requests: List[List[int]] = None,
        generate: bool = False,
        sampling_params: Union[List["SamplingParams"], "SamplingParams", None] = None,
    ):
        if not generate or sampling_params is None:
            sampling_params = SamplingParams(
                temperature=0, prompt_logprobs=1, max_tokens=1, detokenize=False
            )
        if not isinstance(sampling_params, List):
            sampling_params = [sampling_params] * len(requests)
        if self.data_parallel_size > 1 and not self.V1:
            # vLLM hangs if resources are set in ray.remote
            # also seems to only work with decorator and not with ray.remote() fn
            # see https://github.com/vllm-project/vllm/issues/973
            @ray.remote
            def run_inference_one_model(
                model_args: dict,
                sampling_params: List["SamplingParams"],
                requests: List[List[int]],
                lora_request: "LoRARequest",
            ):
                llm = LLM(**model_args)
                return llm.generate(
                    [TokensPrompt(prompt_token_ids=request) for request in requests],
                    sampling_params=sampling_params,
                    lora_request=lora_request,
                )

            # dispatch requests to all self.data_parallel_size workers, in interleaved fashion
            # interleaved important to balance context lengths across workers
            requests = [list(x) for x in distribute(self.data_parallel_size, requests)]
            sampling_params = [
                list(sp) for sp in distribute(self.data_parallel_size, sampling_params)
            ]
            inputs = (
                (self.model_args, sp, req, self.lora_request)
                for req, sp in zip(requests, sampling_params)
            )
            object_refs = [run_inference_one_model.remote(*x) for x in inputs]
            results = ray.get(object_refs)
            # Invoke ray.shutdown() to prevent hang-ups if subsequent calls required.
            ray.shutdown()
            # flatten results
            return undistribute(results)
        elif self.data_parallel_size > 1:
            # based on https://github.com/vllm-project/vllm/blob/a04720bc36401d831cb048c3917b9e58173d9c1d/examples/offline_inference/data_parallel.py
            dp_size = self.data_parallel_size
            dp_master_ip = os.environ.get("VLLM_DP_MASTER_IP", "127.0.0.1")
            dp_master_port = os.environ.get("VLLM_DP_MASTER_PORT") or get_open_port()

            requests = (list(x) for x in distribute(self.data_parallel_size, requests))
            sampling_params = (
                list(sp) for sp in distribute(self.data_parallel_size, sampling_params)
            )
            procs, resq = [], Queue()
            # We use Process as it is non-daemonic
            try:
                for rank, (req, sp) in enumerate(zip(requests, sampling_params)):
                    proc = Process(
                        target=_vllm_mp_worker,
                        args=(
                            self.model_args.copy(),
                            sp,
                            req,
                            self.lora_request,
                            resq,
                            dp_size,
                            rank,
                            dp_master_port,
                            dp_master_ip,
                        ),
                    )
                    proc.start()
                    procs.append(proc)

                # Collect results
                rank_res = {}
                while len(rank_res) < len(procs):
                    try:
                        rank, result = resq.get(timeout=30)
                        if isinstance(result, dict) and "error" in result:
                            raise RuntimeError(result["error"])
                        rank_res[rank] = result
                    except Empty:
                        dead_procs = [
                            idx
                            for idx, p in enumerate(procs)
                            if not p.is_alive() and idx not in rank_res
                        ]
                        if dead_procs:
                            raise RuntimeError(
                                f"Worker processes {dead_procs} died unexpectedly"
                            )
                        continue

                results = [rank_res[i] for i in range(len(procs))]
                return undistribute(results)

            # cleanup
            finally:
                try:
                    resq.close()
                    resq.join_thread()
                except Exception:
                    eval_logger.debug(
                        "Failed to close vllm DP results queue", exc_info=True
                    )
                for proc in procs:
                    proc.join(timeout=10)
                    if proc.is_alive():
                        proc.terminate()
                        proc.join(timeout=5)
                        if proc.is_alive():
                            proc.kill()

        else:
            outputs = self.model.generate(
                [TokensPrompt(prompt_token_ids=request) for request in requests],
                sampling_params=sampling_params,
                use_tqdm=True if self.batch_size == "auto" else False,
                lora_request=self.lora_request,
            )
            return outputs

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

        re_ords = Collator(
            requests,
            _collate_gen,
            group_by=None,
        )
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
                    eval_logger.warning(
                        f"Context length {len(x)} exceeds max length (context + max gen tokens): {max_ctx_len}. Truncating context."
                    )
                    context_encoding_truncated.append(x[-max_ctx_len:])
                else:
                    context_encoding_truncated.append(x)
                # create sampling params
                kwargs = self.modify_gen_kwargs(kwargs)
                sampling_params.append(
                    SamplingParams(max_tokens=max_gen_toks, stop=until, **kwargs)
                )

            # perform batched generation
            cont = self._model_generate(
                requests=context_encoding_truncated,
                generate=True,
                sampling_params=sampling_params,
            )

            # cache generations
            for output, context in zip(cont, context):
                generated_text: str = output.outputs[0].text
                # use secondary stop seqs to cut off should-have-been-stopped content post-hoc
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
                if (
                    full_length := len(context_enc + continuation_enc)
                ) > self.max_length:
                    eval_logger.warning(
                        f"Context length {full_length} exceeds max length ({self.max_length}). Truncating context."
                    )
                inp = (context_enc + continuation_enc)[-(self.max_length) :]
                ctxlen = len(context_enc) - max(
                    0, len(context_enc) + len(continuation_enc) - (self.max_length)
                )

                inputs.append(inp)
                ctxlens.append(ctxlen)

            outputs = self._model_generate(requests=inputs, generate=False)

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
            Contains prompt_logprobs
        :param ctxlen: int
            Length of context (so we can slice them away and only keep the predictions)
        :return:
            continuation_logprobs: float
                Log probabilities of continuation tokens
            is_greedy: bool
                Whether argmax matches given continuation exactly
        """

        # The first entry of prompt_logprobs is None because the model has no previous tokens to condition on.
        continuation_logprobs_dicts = outputs.prompt_logprobs

        def coerce_logprob_to_num(logprob):
            # vLLM changed the return type of logprobs from float
            # to a Logprob object storing the float value + extra data
            # (https://github.com/vllm-project/vllm/pull/3065).
            # If we are dealing with vllm's Logprob object, return
            # the logprob value stored as an attribute. Otherwise,
            # return the object itself (which should be a float
            # for older versions of vLLM).
            return getattr(logprob, "logprob", logprob)

        continuation_logprobs_dicts = [
            {
                token: coerce_logprob_to_num(logprob)
                for token, logprob in logprob_dict.items()
            }
            if logprob_dict is not None
            else None
            for logprob_dict in continuation_logprobs_dicts
        ]

        # Calculate continuation_logprobs
        # assume ctxlen always >= 1
        continuation_logprobs = sum(
            logprob_dict.get(token)
            for token, logprob_dict in zip(
                tokens[ctxlen:], continuation_logprobs_dicts[ctxlen:]
            )
        )

        # Determine if is_greedy
        is_greedy = True
        for token, logprob_dict in zip(
            tokens[ctxlen:], continuation_logprobs_dicts[ctxlen:]
        ):
            # Get the token with the maximum log probability from the logprob_dict
            if logprob_dict:  # Ensure the logprob_dict is not None
                top_token = max(logprob_dict, key=logprob_dict.get)
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
