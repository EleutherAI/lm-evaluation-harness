import math
import torch
import torch.nn.functional as F
import transformers
from typing import List, Mapping, NewType, Optional, Tuple, Union
from tqdm import tqdm

from transformers import BatchEncoding

from lm_eval import utils
from lm_eval.base import BaseLM

TokenSequence = Union[List[int], torch.LongTensor, torch.Tensor, BatchEncoding]

_DeviceMapping = NewType("DeviceMapping", Mapping[str, Union[int, str, torch.device]])


def _get_accelerate_args(
    device_map_option: Optional[str] = "auto",
    max_memory_per_gpu: Optional[Union[int, str]] = None,
    max_cpu_memory: Optional[Union[int, str]] = None,
    offload_folder: Optional[str] = "./offload",
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
    args["device_map"] = device_map_option
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


class HuggingFaceAutoLM(BaseLM):
    AUTO_CONFIG_CLASS: transformers.AutoConfig = transformers.AutoConfig
    AUTO_TOKENIZER_CLASS: transformers.AutoTokenizer = transformers.AutoTokenizer
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
        add_special_tokens: Optional[bool] = None,
        use_accelerate: Optional[bool] = False,
        device_map_option: Optional[str] = "auto",
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[str] = "./offload",
        dtype: Optional[Union[str, torch.dtype]] = None,
        device: Optional[Union[int, str]] = "cuda",
        trust_remote_code: Optional[bool] = False,
    ):
        """Initializes a HuggingFace `AutoModel` and `AutoTokenizer` for evaluation.
        Args:
            pretrained (str):
                The HuggingFace Hub model ID name or the path to a pre-trained
                model to load. This is effectively the `pretrained_model_name_or_path`
                argument of `from_pretrained` in the HuggingFace `transformers` API.
            add_special_tokens (bool, optional, defaults to True):
                Whether to add special tokens to the input sequences. If `None`, the
                default value will be set to `True` for seq2seq models (e.g. T5) and
                `False` for causal models.
                WARNING: Evaluating causal models with `add_special_tokens=True` is
                currently __not__ supported.
            > Large model loading `accelerate` arguments
            use_accelerate (bool, optional, defaults to False):
                If True, uses the `accelerate` library to load a large model across
                multiple devices.
            device_map_option (str, optional, defaults to "auto"):
                The device map option to use when loading the model with
                `accelerate`.
                Options:
                    "auto", "balanced", "balanced_low_0", "sequential"
                See the `accelerate` docs for more details on these options:
                https://huggingface.co/docs/accelerate/v0.12.0/en/usage_guides/big_modeling#designing-a-device-map
            max_memory_per_gpu (Union[int, str], optional, defaults to None):
                The maximum memory available for each GPU in bytes as `int` or in
                the format f"{significand}{unit_symbol}" where {unit_symbol} is
                any of ["GB", "MB", "GIB", "MIB"]. Refer to the `max_memory` arg in
                the "Parameters for big model inference" section of the following
                docs:
                https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/model#large-model-loading
            max_cpu_memory (Union[int, str], optional, defaults to None):
                The maximum available CPU RAM in bytes as `int` or in the format
                f"{significand}{unit_symbol}" where {unit_symbol} is any of
                ["GB", "MB", "GIB", "MIB"]. Refer to the `max_memory` arg in the
                "Parameters for big model inference" section of the following docs:
                https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/model#large-model-loading
            offload_folder (str, optional, defaults to "./offload"):
                The folder to offload weights into if `device_map` contains any
                "disk" value.
            dtype (Union[str, torch.dtype], optional, defaults to None):):
                Converts the model weights to `dtype`, if specified. Strings get
                converted to `torch.dtype` objects (e.g. `float16` -> `torch.float16`).
                Use `dtype="auto"` to derive the type from the model’s weights.
        """
        super().__init__()

        assert isinstance(pretrained, str)
        assert isinstance(device, str)
        assert isinstance(batch_size, int)
        if (
            add_special_tokens is not None
            and self.AUTO_MODEL_CLASS is transformers.AutoModelForCausalLM
        ):
            # TODO: Support evaluating causal models with special tokens. Currently,
            # this is not possible because the `_loglikelihood_tokens()` method for
            # causal LMs makes a no-special-tokens assumption given that contexts
            # and labels/continuations are tokenized separately without special
            # tokens, concatenated, and then processed as inputs.
            assert (
                not add_special_tokens
            ), "Evaluating causal models with `add_special_tokens=True` is currently not supported."

        self._batch_size = batch_size  # TODO: Adaptive batch size
        self._max_gen_toks = max_gen_toks
        self._max_length = max_length
        self._trust_remote_code = trust_remote_code
        self._config = self.AUTO_CONFIG_CLASS.from_pretrained(
            pretrained,
            revision=revision + ("/" + subfolder if subfolder is not None else ""),
            trust_remote_code=self._trust_remote_code,
        )

        self._add_special_tokens = add_special_tokens
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
                device_map_option,
                max_memory_per_gpu,
                max_cpu_memory,
                offload_folder,
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
            trust_remote_code=self._trust_remote_code,
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
        tokenizer = self.AUTO_TOKENIZER_CLASS.from_pretrained(
            pretrained if tokenizer is None else tokenizer,
            revision=revision + ("/" + subfolder if subfolder is not None else ""),
            trust_remote_code=self._trust_remote_code,
        )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @property
    def add_special_tokens(self) -> bool:
        """Whether to include special tokens in encoded text. This should be
        determined by whether or not the model was trained with special tokens.
        TODO: Remove these conditionals once HuggingFace supports a way to
        check whether or not an arbitrary model was trained with special tokens.
        """
        if self._add_special_tokens is not None:
            return self._add_special_tokens
        elif self.AUTO_MODEL_CLASS is transformers.AutoModelForCausalLM:
            return False
        elif self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM:
            return True
        else:
            raise ValueError(
                "Could not determine `add_special_tokens` value from the model "
                "class. Set to `True` or `False` depending on whether the model "
                "was pre-trained with special tokens."
            )

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
        return self.tokenizer.encode(string, add_special_tokens=self.add_special_tokens)

    def tok_encode_batch(self, strings: List[str]) -> TokenSequence:
        return self.tokenizer(
            strings,
            padding=True,
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt",
        )

    def tok_decode(self, tokens: torch.LongTensor) -> List[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def greedy_until(self, requests: List[Tuple[str, dict]]) -> List[str]:
        def _collate(x):
            tokens = self.tok_encode(x[0])
            return len(tokens), x[0]

        results = []
        reorder = utils.Reorderer(requests, _collate)
        for chunk in utils.chunks(tqdm(reorder.get_reordered(), disable=False), self.batch_size):
            context = [c[0] for c in chunk]
            until = chunk[0][1]
            for c in chunk:
                assert c[1] == until, \
                    "`until` condition must be the same across batch elements (%s vs %s). Use batch_size=1." % (
                        c[1], until
                    )
            max_tokens = self.max_gen_toks
            token_context = self.tok_encode_batch(context)
            generated_tokens = self._model_generate(
                inputs=token_context,
                max_tokens=max_tokens,
                stop=until,
            )
            generated_texts = self.postprocess(
                generated_tokens=generated_tokens,
                prefix_length=token_context['input_ids'].size(1),
                until=until,
                is_greedy=True
            )
            for text in generated_texts:
                self.cache_hook.add_partial("greedy_until", (context, until), text)
            results.extend(generated_texts)

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
        self,
        inputs: transformers.BatchEncoding,
        max_tokens: int,
        stop: Optional[List[str]] = None,
        num_return_sequences: int = 1,
        num_return_sequences_batch: int = -1,
        temperature: float = 0.0
    ) -> TokenSequence:

        if isinstance(stop, str):
            stop = [stop]

        input_ids = inputs["input_ids"][:, self.max_gen_toks-self.max_length:]
        attention_mask = inputs["attention_mask"][:, self.max_gen_toks-self.max_length:]

        # We only support batching over examples when `num_return_sequences is 1`
        # (see todo in `.generate()`.
        assert (input_ids.size(0) == 1 or num_return_sequences == 1)

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        if num_return_sequences_batch > 0:
            num_batches = num_return_sequences // num_return_sequences_batch
            num_return_sequences = num_return_sequences_batch
        else:
            num_batches = 1

        generated_tokens = []
        for _ in range(num_batches):
            stopping_criteria = stop_sequences_criteria(
                self.tokenizer, stop, input_ids.shape[1],
                batch_size=input_ids.shape[0]*num_return_sequences
            )
            generated_tokens_ = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                stopping_criteria=stopping_criteria,
                do_sample=temperature > 0,
                temperature=temperature,
                num_return_sequences=num_return_sequences
            )
            generated_tokens.append(generated_tokens_)
        generated_tokens = self._pad_and_combine(generated_tokens)
        return generated_tokens

    def generate(self, requests):
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return len(toks), x[0]

        re_ord = utils.Reorderer(requests, _collate)

        # TODO: batching along example dimension (in addition to the current batching
        # along the `num_return_sequences` dimension). This would involve
        # iterating over chunks (`utils.chunks`) and modifying `._model_generate`.
        for request in tqdm(re_ord.get_reordered()):
            context, until, is_greedy, _model_generate_kwargs = self.parse_request(request)

            context_enc = self.tok_encode_batch(context)
            generated_tokens = self._model_generate(
                inputs=context_enc,
                max_tokens=self.max_gen_toks,
                stop=until,
                **_model_generate_kwargs
            )
            generated_texts = self.postprocess(
                generated_tokens=generated_tokens,
                prefix_length=context_enc['input_ids'].shape[1],
                until=until,
                is_greedy=is_greedy
            )
            cache = (context, until, tuple(_model_generate_kwargs))
            self.cache_hook.add_partial("generate", cache, generated_texts)
            res.append(generated_texts)
        print(re_ord.get_original(res))
        return re_ord.get_original(res)

    def postprocess(self, generated_tokens, prefix_length, until, is_greedy):
        generated_tokens = generated_tokens[:, prefix_length:]
        generated_texts = self.tok_decode(generated_tokens)
        for term in until:
            generated_texts = [candidate.split(term)[0] for candidate in generated_texts]
        return generated_texts

    def _pad_and_combine(self, list_of_generated_tokens):
        # Pad the generated vectors such that they have the same length
        max_length = max(element.size(1) for element in list_of_generated_tokens)
        padded_vectors = []
        for vector in list_of_generated_tokens:
            if vector.size(1) < max_length:
                vector = torch.cat([
                    vector,
                    torch.zeros(
                        vector.size(0), max_length - vector.size(1),
                        dtype=vector.dtype,
                        device=vector.device
                    ).fill_(self.tokenizer.pad_token_id)], dim=1
                )
            padded_vectors.append(vector)
        return torch.cat(padded_vectors, dim=0)


class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(
        self,
        sequence: str,
        tokenizer: transformers.PreTrainedTokenizer,
        initial_decoder_input_length: int,
        batch_size: int,
    ):
        self.initial_decoder_input_length = initial_decoder_input_length
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
        self.sequence_id_len = len(self.sequence_ids)
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
        lookback_ids_batch = input_ids[:, self.initial_decoder_input_length :][
            :, -self.sequence_id_len :
        ]

        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)

        for i, done in enumerate(self.done_tracker):
            if not done:
                self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker


def stop_sequences_criteria(
    tokenizer: transformers.PreTrainedTokenizer,
    stop_sequences: List[str],
    initial_decoder_input_length: int,
    batch_size: int,
) -> transformers.StoppingCriteriaList:
    return transformers.StoppingCriteriaList(
        [
            *[
                MultiTokenEOSCriteria(
                    sequence, tokenizer, initial_decoder_input_length, batch_size
                )
                for sequence in stop_sequences
            ],
        ]
    )

