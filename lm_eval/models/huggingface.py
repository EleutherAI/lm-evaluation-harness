import math
import torch
import torch.nn.functional as F
import transformers
from typing import List, Mapping, NewType, Optional, Tuple, Union
from tqdm import tqdm

from lm_eval.api import utils
from lm_eval.api.model import TokenLM, TokenSequence


_DeviceMapping = NewType("DeviceMapping", Mapping[str, Union[int, str, torch.device]])


def _get_accelerate_args(
    max_memory_per_gpu: Optional[Union[int, str]],
    max_cpu_memory: Optional[Union[int, str]],
    offload_folder: Optional[str],
) -> dict:
    """Returns the kwargs needed to apply `accelerate` in `AutoModel.from_pretrained`."""
    max_memory = {}
    if max_memory_per_gpu is not None:
        max_memory_per_gpu_map = {
            device_idx: max_memory_per_gpu
            for device_idx in range(torch.cuda.device_count())
        }
        max_memory.update(max_memory_per_gpu_map)
    if max_cpu_memory is not None:
        max_memory["cpu"] = max_cpu_memory

    args = {}
    if max_memory:
        args["max_memory"] = max_memory
    args["device_map"] = "auto"
    args["offload_folder"] = offload_folder
    return args


def _get_dtype(
    dtype: Union[str, torch.dtype], config: Optional[transformers.AutoConfig] = None
) -> torch.dtype:
    """Converts `dtype` from `str` to torch.dtype when possible."""
    if dtype is None and config is not None:
        _torch_dtype = config.torch_dtype
    elif isinstance(dtype, str) and dtype != "auto":
        # Convert `str` args torch dtype: `float16` -> `torch.float16`
        _torch_dtype = getattr(torch, dtype)
    else:
        _torch_dtype = dtype
    return _torch_dtype


class HuggingFaceAutoLM(TokenLM):

    AUTO_MODEL_CLASS: transformers.AutoModel = None

    # Default max sequence length setting for when no `max_length` is provided
    # or no max length config setting is found in the model or tokenizer.
    _DEFAULT_MAX_LENGTH: int = 2048

    def __init__(
        self,
        pretrained: str,
        tokenizer: Optional[str] = None,
        subfolder: Optional[str] = None,
        revision: Optional[str] = "main",
        batch_size: Optional[int] = 1,
        max_gen_toks: Optional[int] = 256,
        max_length: Optional[int] = None,
        use_accelerate: Optional[bool] = False,
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[str] = "./offload",
        dtype: Optional[Union[str, torch.dtype]] = None,
        device: Optional[Union[int, str]] = "cuda",
    ):
        """Initializes a HuggingFace `AutoModel` and `AutoTokenizer` for evaluation.

        :param use_accelerate:
            If True, uses the `accelerate` library to load a large model across
            multiple devices.
        :param max_memory_per_gpu: Optional[Union[int, str]]
            The maximum memory available for each GPU in bytes as `int` or in
            the format f"{significand}{unit_symbol}" where {unit_symbol} is
            any of ["GB", "MB", "GIB", "MIB"]. Refer to the `max_memory` arg in
            the "Parameters for big model inference" section of the following docs:
            https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/model#large-model-loading
        :param max_cpu_memory: Optional[Union[int, str]]
            The maximum available CPU RAM in bytes as `int` or in the format
            f"{significand}{unit_symbol}" where {unit_symbol} is any of
            ["GB", "MB", "GIB", "MIB"]. Refer to the `max_memory` arg in the
            "Parameters for big model inference" section of the following docs:
            https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/model#large-model-loading
        :param offload_folder: Optional[str]
            The folder to offload weights into if `device_map` contains any "disk" value.
        :param dtype: Optional[Union[str, torch.dtype]]
            Converts the model weights to `dtype`, if specified. Strings get
            converted to `torch.dtype` objects (e.g. `float16` -> `torch.float16`).
            Use `dtype="auto"` to derive the type from the model’s weights.
        """
        super().__init__()

        assert isinstance(pretrained, str)
        assert isinstance(device, str)
        assert isinstance(batch_size, int)

        self._batch_size = batch_size  # TODO: Adaptive batch size
        self._max_gen_toks = max_gen_toks
        self._max_length = max_length
        self._config = transformers.AutoConfig.from_pretrained(pretrained)

        self.tokenizer = self._create_auto_tokenizer(
            pretrained=pretrained,
            revision=revision,
            subfolder=subfolder,
            tokenizer=tokenizer,
        )
        self.tokenizer.model_max_length = self.max_length

        accelerate_kwargs = {}
        if use_accelerate:
            accelerate_kwargs = _get_accelerate_args(
                max_memory_per_gpu, max_cpu_memory, offload_folder
            )
        self.model = self._create_auto_model(
            pretrained=pretrained,
            revision=revision,
            subfolder=subfolder,
            torch_dtype=_get_dtype(dtype, self._config),
            **accelerate_kwargs,
        )
        self.model.eval()
        torch.set_grad_enabled(False)

        self._device = device
        if use_accelerate and "lm_head" in self.model.hf_device_map:
            # `accelerate` can place `lm_head` weights on a different device than
            # the user specified one so we force `self._device` to be the same as
            # `lm_head`'s.
            self._device = self.model.hf_device_map["lm_head"]
        if not use_accelerate:
            self.model.to(self._device)

    def _create_auto_model(
        self,
        *,
        pretrained: str,
        revision: str,
        subfolder: str,
        device_map: Optional[Union[str, _DeviceMapping]] = None,
        max_memory: Optional[dict] = None,
        offload_folder: Optional[str] = None,
        torch_dtype: Optional[Union[str, torch.dtype]] = None,
    ) -> transformers.AutoModel:
        """Returns a pre-trained pytorch model from a pre-trained model configuration."""
        model = self.AUTO_MODEL_CLASS.from_pretrained(
            pretrained,
            revision=revision + ("/" + subfolder if subfolder is not None else ""),
            device_map=device_map,
            max_memory=max_memory,
            offload_folder=offload_folder,
            torch_dtype=torch_dtype,
        )
        return model

    def _create_auto_tokenizer(
        self,
        *,
        pretrained: str,
        revision: str,
        subfolder: str,
        tokenizer: Optional[str] = None,
    ) -> transformers.PreTrainedTokenizer:
        """Returns a pre-trained tokenizer from a pre-trained tokenizer configuration."""
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained if tokenizer is None else tokenizer,
            revision=revision,
            subfolder=subfolder,
        )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @property
    def eot_token(self) -> str:
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def max_length(self) -> int:
        """Return the maximum sequence length of the model.
        NOTE: Different model configurations have different max sequence length
        attribute names.
            - n_positions: (CTRLConfig)
            - max_position_embeddings: (BartConfig, RoFormerConfig)
            - n_ctx: (GPT2Config)
        NOTE: For relative position encoded models you should specify the max
        sequence length of the model in the constructor via `max_length`.
        """
        if self._max_length is not None:
            return self._max_length
        # Try to get the sequence length from the model config.
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self._config, attr):
                return getattr(self._config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def batch_size(self) -> int:
        # TODO: Add adaptive batch size.
        return self._batch_size  # * gpus

    @property
    def device(self) -> Union[int, str, torch.device]:
        return self._device

    def tok_encode(self, string: str) -> TokenSequence:
        # TODO: Merge `tok_encode_batch` here.
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_encode_batch(self, strings: List[str]) -> TokenSequence:
        return self.tokenizer(
            strings, padding=True, add_special_tokens=False, return_tensors="pt"
        )

    def tok_decode(self, tokens: torch.LongTensor) -> List[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def greedy_until(self, requests: List[Tuple[str, dict]]) -> List[str]:
        def _collate(x):
            tokens = self.tok_encode(x[0])
            return len(tokens), x[0]

        results = []
        reorder = utils.Reorderer(requests, _collate)
        for chunk in utils.chunks(
            tqdm(reorder.get_reordered(), disable=False), self.batch_size
        ):
            context = [c[0] for c in chunk]
            request_args = chunk[0][1]
            stop_sequences = request_args["stop_sequences"]
            max_generation_length = request_args["max_generation_length"]
            num_fewshot = request_args["num_fewshot"]

            assert (
                isinstance(max_generation_length, int) or max_generation_length is None
            )
            assert isinstance(stop_sequences, list) or stop_sequences is None
            assert isinstance(num_fewshot, int) or num_fewshot is None

            # TODO: Find a better way to handle stop sequences for 0-shot.
            if stop_sequences is None or num_fewshot == 0:
                until = [self.eot_token]
            else:
                until = stop_sequences + [self.eot_token]

            if max_generation_length is None:
                max_tokens = self.max_gen_toks
            else:
                max_tokens = max_generation_length

            # Ensure that the context does not encroach into the `space`
            # for the generation.
            token_context = self.tok_encode_batch(context)
            input_ids = token_context["input_ids"][
                :, self.max_gen_toks - self.max_length :
            ].to(self.device)
            attention_mask = token_context["attention_mask"][
                :, self.max_gen_toks - self.max_length :
            ].to(self.device)

            responses = self._model_generate(
                inputs={"input_ids": input_ids, "attention_mask": attention_mask},
                max_tokens=max_tokens,
                stop=until,
            )
            responses = self.tok_decode(responses.tolist())

            for response in responses:
                for term in until:
                    response = response.split(term)[0]
                # partial caching
                self.cache_hook.add_partial("greedy_until", (context, until), response)
                results.append(response)
        return reorder.get_original(results)


class AutoCausalLM(HuggingFaceAutoLM):
    """Causal language modeling.
    You can find a set of supported models in the HF documentation:
    https://huggingface.co/docs/transformers/main/model_doc/auto#transformers.AutoModelForCausalLM
    """

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def _create_auto_tokenizer(
        self,
        *,
        pretrained: str,
        revision: str,
        subfolder: str,
        tokenizer: Optional[str] = None,
    ) -> transformers.PreTrainedTokenizer:
        tokenizer = super()._create_auto_tokenizer(
            pretrained=pretrained,
            revision=revision,
            subfolder=subfolder,
            tokenizer=tokenizer,
        )
        tokenizer.padding_side = "left"
        return tokenizer

    def _model_call(
        self, inputs: TokenSequence, labels: Optional[TokenSequence] = None
    ) -> TokenSequence:
        return self.model(inputs)["logits"]

    def _model_generate(
        self, inputs: TokenSequence, max_tokens: int, stop: Optional[List[str]] = None
    ) -> TokenSequence:
        stopping_criteria = stop_sequences_criteria(self.tokenizer, stop)
        generations = self.model.generate(
            **inputs,
            # GPT style models require the `generate` `max_length` arg to include the
            # context length, so we instead set `max_new_tokens` which is the number
            # of new tokens to generate, excluding the current number of tokens.
            max_new_tokens=max_tokens,
            stopping_criteria=stopping_criteria,
            do_sample=False,
        )
        return utils.select_continuation_from_batch_left_padding(
            generations, max_context_size=inputs["input_ids"].size(1)
        )


class AutoSeq2SeqLM(HuggingFaceAutoLM):
    """Seq2Seq language modeling.
    You can find a set of supported models in the following documentation:
    https://huggingface.co/docs/transformers/main/model_doc/auto#transformers.AutoModelForSeq2SeqLM
    """

    AUTO_MODEL_CLASS = transformers.AutoModelForSeq2SeqLM

    @property
    def max_length(self) -> int:
        """Return the maximum sequence length of the model.
        TODO: Currently only works for relative position encoded Seq2Seq models.
        """
        if self._max_length is not None:
            return self._max_length
        return self._DEFAULT_MAX_LENGTH

    def loglikelihood(
        self, requests: List[Tuple[str, str]]
    ) -> List[Tuple[float, bool]]:
        new_requests = []
        for chunk in utils.chunks(requests, self.batch_size):
            context, continuation = zip(*chunk)
            # Fill empty contexts with the EOT token.
            context = [
                f"{self.eot_token}" if len(text) == 0 else text for text in context
            ]
            context_enc = self.tok_encode_batch(context)
            for key in context_enc:
                context_enc[key] = context_enc[key][:, -self.max_length :]
            continuation_enc = self.tok_encode_batch(list(continuation))
            for key in continuation_enc:
                continuation_enc[key] = continuation_enc[key][:, -self.max_length :]
            new_requests.append(
                ((context, continuation), context_enc, continuation_enc)
            )
        return self._loglikelihood_tokens(new_requests)

    def loglikelihood_rolling(self, requests: List[Tuple[str, str]]) -> List[float]:
        loglikelihoods = []
        for (string,) in tqdm(requests):
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
            contexts, conts = utils.split_and_pad_windows(
                rolling_token_windows,
                pad_token_id=self.eot_token_id,
                max_seq_len=self.max_length,
            )
            # Manually create BatchEncoding tensors with attention masks as
            # expected by `self._model_call` in `self._loglikelihood_tokens`.
            contexts_enc = torch.Tensor(contexts).long()
            contexts_enc = transformers.tokenization_utils_base.BatchEncoding(
                {
                    "input_ids": contexts_enc,
                    "attention_mask": (contexts_enc != self.eot_token_id).long(),
                }
            )
            conts_enc = torch.Tensor(conts).long()
            conts_enc = transformers.tokenization_utils_base.BatchEncoding(
                {
                    "input_ids": conts_enc,
                    "attention_mask": (conts_enc != self.eot_token_id).long(),
                }
            )
            # TODO: Extract out this call so it only gets called once and also
            # somehow figure out partial caching for.
            rolling_token_windows_request = [
                ((contexts, conts), contexts_enc, conts_enc)
            ]
            string_nll = self._loglikelihood_tokens(
                rolling_token_windows_request, disable_tqdm=True
            )
            string_nll = [x[0] for x in string_nll]  # discard is_greedy
            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)
        return loglikelihoods

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], TokenSequence, TokenSequence]],
        disable_tqdm: Optional[bool] = False,
    ) -> List[Tuple[float, bool]]:
        results = []
        for chunk in tqdm(
            requests, total=math.ceil(len(requests)), disable=disable_tqdm
        ):
            cache_keys, inputs_tokens, targets_tokens = chunk
            inputs_tokens = inputs_tokens.to(self.device)
            targets_tokens = targets_tokens.to(self.device)
            outputs = self._model_call(inputs=inputs_tokens, labels=targets_tokens)
            log_softmaxes = F.log_softmax(outputs.logits, dim=-1)

            output_iterator = zip(
                zip(cache_keys[0], cache_keys[1]),
                log_softmaxes,
                targets_tokens["input_ids"],
                targets_tokens["attention_mask"],
            )
            for cache_key, log_softmax, target_tokens, target_mask in output_iterator:
                length = target_mask.sum()
                print("target_mask", target_mask)
                print("logit size:", log_softmax.shape)
                log_softmax = log_softmax[:length]
                print("truncated logit size:", log_softmax.shape)
                target_tokens = target_tokens[:length]
                greedy_tokens = log_softmax.argmax(dim=-1)
                max_equal = (greedy_tokens == target_tokens).all()
                target_logits = torch.gather(
                    log_softmax, 1, target_tokens.unsqueeze(-1)
                ).squeeze(-1)
                answer = (float(target_logits.sum()), bool(max_equal))
                results.append(answer)
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)
        return results

    def _model_call(
        self, inputs: TokenSequence, labels: Optional[TokenSequence] = None
    ) -> TokenSequence:
        return self.model(**inputs, labels=labels["input_ids"])

    def _model_generate(
        self, inputs: TokenSequence, max_tokens: int, stop: Optional[List[str]] = None
    ) -> Union[TokenSequence, List[str]]:
        stopping_criteria = stop_sequences_criteria(self.tokenizer, stop)
        generations = self.model.generate(
            **inputs,
            max_length=max_tokens,
            stopping_criteria=stopping_criteria,
            do_sample=False,
        )
        return generations


class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(self, sequence: str, tokenizer: transformers.PreTrainedTokenizer):
        self.sequence = sequence
        self.sequence_id = tokenizer.encode(sequence)
        self.sequence_id_len = len(self.sequence_id) + 1
        self.tokenizer = tokenizer

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        last_token_id = input_ids[0, -self.sequence_id_len :]
        last_tokens = self.tokenizer.decode(last_token_id)
        is_stopped = self.sequence in last_tokens
        return is_stopped


def stop_sequences_criteria(
    tokenizer: transformers.PreTrainedTokenizer, stop_sequences: List[str]
) -> transformers.StoppingCriteriaList:
    return transformers.StoppingCriteriaList(
        [
            *[
                MultiTokenEOSCriteria(sequence, tokenizer)
                for sequence in stop_sequences
            ],
        ]
    )
