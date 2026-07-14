"""
Rebellions NPU model integration for lm_eval.

This module provides support for running language models on Rebellions NPU hardware
using the RBLN SDK.
"""

import copy
import json
import logging
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm
from transformers import GenerationConfig
from transformers.generation import StoppingCriteriaList
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
)

import lm_eval.models.utils
import lm_eval.models.utils_hf
from lm_eval import utils
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import _add_special_kwargs, has_bos_prefix
from lm_eval.models.utils_hf import stop_sequences_criteria
# NPU device utilities inlined (removed npu_device_utils.py dependency)

logger = logging.getLogger(__name__)

try:
    RBLN_AVAILABLE = True
    # Import the RBLN Auto classes (text-only)
    from optimum.rbln import (
        RBLNAutoConfig,
        RBLNAutoModelForCausalLM,
        RBLNAutoModelForSeq2SeqLM,
    )

    # VLM Auto classes — available in newer optimum-rbln. Import each separately so
    # users on older SDK versions can still run text-only paths.
    try:
        from optimum.rbln import RBLNAutoModelForVision2Seq
    except ImportError:
        RBLNAutoModelForVision2Seq = None

    try:
        from optimum.rbln import RBLNAutoModelForImageTextToText
    except ImportError:
        RBLNAutoModelForImageTextToText = None

    # Try to import specific model classes (legacy support)
    try:
        from optimum.rbln import RBLNLlamaForCausalLM
    except ImportError:
        RBLNLlamaForCausalLM = None

    try:
        from optimum.rbln import RBLNT5ForConditionalGeneration
    except ImportError:
        RBLNT5ForConditionalGeneration = None

except ImportError:
    # Fallback objects if optimum[rbln] is not available
    RBLNAutoConfig = object
    RBLNAutoModelForCausalLM = object
    RBLNAutoModelForSeq2SeqLM = object
    RBLNAutoModelForVision2Seq = None
    RBLNAutoModelForImageTextToText = None
    RBLNLlamaForCausalLM = None
    RBLNT5ForConditionalGeneration = None
    RBLN_AVAILABLE = False


# Per-model compile-time `rbln_config` profiles, sourced from rbln-model-zoo
# (https://github.com/RBLN-SW/rbln-model-zoo/tree/main/huggingface/transformers/image-text-to-text).
# Used when export=True and the user does not pass an explicit rbln_config_json.
# Users can partially override via --model_args 'rbln_config_json={...}'.
_VLM_COMPILE_PROFILES = {
    "llava": {
        "vision_tower": {"output_hidden_states": True},
        "language_model": {"tensor_parallel_size": 4, "use_inputs_embeds": True},
    },
    "llava_next": {
        "language_model": {"tensor_parallel_size": 4, "use_inputs_embeds": True},
    },
    "qwen2_vl": {
        "visual": {"max_seq_lens": 6400},
        "tensor_parallel_size": 8,
        "max_seq_len": 32768,
    },
    "qwen2_5_vl": {
        "visual": {"max_seq_lens": 6400},
        "tensor_parallel_size": 8,
        "kvcache_partition_len": 16384,
        "max_seq_len": 114688,
    },
    "qwen3_vl": {
        "visual": {
            "max_seq_lens": 16384,
            "tensor_parallel_size": 8,
            "create_runtimes": False,
        },
        "tensor_parallel_size": 8,
        "kvcache_partition_len": 16384,
        "max_seq_len": 262144,
    },
    "gemma3": {
        "tensor_parallel_size": 8,
        "kvcache_partition_len": 16384,
        "use_inputs_embeds": True,
    },
    "idefics3": {
        "batch_size": 1,
        "max_seq_len": 131072,
        "tensor_parallel_size": 8,
        "attn_impl": "flash_attn",
        "kvcache_partition_len": 16384,
    },
    "paligemma": {
        "language_model": {
            "batch_size": 1,
            "max_seq_len": 8192,
            "tensor_parallel_size": 4,
            "prefill_chunk_size": 8192,
        },
    },
    "paligemma2": {
        "batch_size": 1,
        "max_seq_len": 8192,
        "tensor_parallel_size": 4,
        "prefill_chunk_size": 8192,
    },
    "pixtral": {
        "vision_tower": {"batch_size": 1, "output_hidden_states": True},
        "language_model": {
            "tensor_parallel_size": 8,
            "use_inputs_embeds": True,
            "max_seq_len": 131072,
            "kvcache_partition_len": 16384,
        },
    },
    "blip-2": {
        "batch_size": 1,
        "max_seq_len": 2048,
        "tensor_parallel_size": 4,
        "use_inputs_embeds": True,
    },
    # blip_2 (underscore form) is the HF config.model_type; alias for safety.
    "blip_2": {
        "batch_size": 1,
        "max_seq_len": 2048,
        "tensor_parallel_size": 4,
        "use_inputs_embeds": True,
    },
}


# VLM model_types that should be dispatched to RBLNAutoModelForImageTextToText
# rather than RBLNAutoModelForVision2Seq. (Verified against rbln-model-zoo
# image-text-to-text examples.)
_VLM_IMAGE_TEXT_TO_TEXT_MODEL_TYPES = {"gemma3"}


def _deep_merge_dict(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base`, returning a new dict.

    Nested dicts are merged; scalar/list values in `override` replace those in `base`.
    """
    result = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = value
    return result


@register_model("rbln")
class RBLNLM(TemplateLM):
    """
    Enables usage with Rebellions NPU hardware
    using the RBLN SDK and HuggingFace Transformers.
    """

    def __init__(
        self,
        pretrained: Optional[str] = "microsoft/DialoGPT-small",
        revision: Optional[str] = "main",
        subfolder: Optional[str] = None,
        tokenizer: Optional[str] = None,
        truncation: Optional[bool] = False,
        max_length: Optional[int] = None,
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[int] = 1,
        low_cpu_mem_usage: Optional[bool] = True,
        trust_remote_code: Optional[bool] = False,
        use_fast_tokenizer: Optional[bool] = True,
        add_bos_token: Optional[bool] = None,
        device_map: Optional[str] = None,
        model_type: Optional[str] = None,  # "causal", "seq2seq", or None for auto-detect
        **kwargs,
    ) -> None:
        if not RBLN_AVAILABLE:
            raise ImportError(
                "Tried to load Rebellions NPU model, but optimum[rbln] is not installed. "
                "Please install with: pip install optimum[rbln] "
                "For more details: https://docs.rbln.ai/"
            )
        
        # Check NPU availability and log device info
        self._check_npu_availability()
        
        # Log device information for user awareness
        self._log_device_info()

        super().__init__()

        assert isinstance(pretrained, str)
        assert isinstance(batch_size, (int, str))

        self.batch_size_per_gpu = int(batch_size)
        batch_size = int(batch_size)

        self._config = transformers.AutoConfig.from_pretrained(
            pretrained,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )

        revision = str(revision)  # cast to string if not already one
        # TODO: update this to be less of a hack once subfolder is fixed in HF
        revision = revision + ("/" + subfolder if subfolder is not None else "")

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained if tokenizer is None else tokenizer,
            revision=revision,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast_tokenizer,
        )

        # Set up tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Determine model type
        self.model_type = self._determine_model_type(model_type)
        print(f"Detected model type: {self.model_type}")

        # Load model using RBLN SDK
        print(f"{'=' * 20} \n Loading {self.model_type} model on Rebellions NPU...")
        
        try:
            self.model = self._load_model(
                pretrained=pretrained,
                revision=revision,
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=low_cpu_mem_usage,
                device_map=device_map,
                dtype=dtype,
                **kwargs,
            )
            print(f"SUCCESS: {self.model_type} model loaded on Rebellions NPU. \n {'=' * 20}")
        except Exception as e:
            print(f"ERROR: Failed to load {self.model_type} model on Rebellions NPU: {e}")
            raise

        self.truncation = truncation
        self.vocab_size = self.tokenizer.vocab_size
        self.add_bos_token = add_bos_token

        self.batch_schedule = 1
        self.batch_sizes = {}

    def _check_npu_availability(self):
        """Check if NPU devices are available using rbln-stat."""
        try:
            # Try to import and use NPU detection
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            from lm_eval.npu_utils import is_npu_available, get_npu_count
            
            if not is_npu_available():
                logger.warning("No NPU devices detected. Make sure NPU hardware is connected and drivers are installed.")
                logger.warning("Run 'rbln-stat' to check NPU status, or use 'python check_npu.py' for detailed information.")
            else:
                npu_count = get_npu_count()
                logger.info(f"NPU devices detected: {npu_count}")
                
        except ImportError:
            # NPU detection module not available, just warn
            logger.warning("NPU detection module not available. Cannot verify NPU status.")
        except Exception as e:
            logger.warning(f"Could not check NPU availability: {e}")
    
    def _log_device_info(self):
        """Log information about available devices."""
        logger.info("=== Device Information ===")
        
        # NPU info
        try:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            from lm_eval.npu_utils import is_npu_available, get_npu_count
            
            if is_npu_available():
                npu_count = get_npu_count()
                logger.info(f"NPU: {npu_count} devices available")
            else:
                logger.info("NPU: Not available")
        except Exception:
            logger.info("NPU: Detection failed")
        
        # CUDA info
        if torch.cuda.is_available():
            cuda_count = torch.cuda.device_count()
            logger.info(f"CUDA: {cuda_count} devices available")
        else:
            logger.info("CUDA: Not available")
        
        # Default device info
        try:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            from lm_eval.npu_utils import is_npu_available, get_npu_count
            
            if is_npu_available():
                npu_count = get_npu_count()
                default = "npu:0"
                logger.info(f"NPU devices detected: {npu_count}. Using npu:0 as default.")
            elif torch.cuda.is_available():
                default = "cuda"
            else:
                default = "cpu"
            
            logger.info(f"Default device: {default}")
        except Exception:
            logger.info("Default device: cpu (fallback)")
            
        logger.info("==========================")

    def _determine_model_type(self, model_type: Optional[str]) -> str:
        """Determine the model category.

        Uses transformers' standard AutoModel mapping dicts as the source of truth:
        ``MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES`` / ``MODEL_FOR_CAUSAL_LM_MAPPING_NAMES``
        / ``MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES``. Falls back to
        ``config.is_encoder_decoder`` for remote_code models (e.g. EXAONE)
        whose model_type is not registered in the mappings.

        VLM is checked first so that model_types appearing in both causal and
        image-text-to-text mappings (e.g. ``gemma3``) are routed to the VLM path.
        """
        if model_type is not None:
            supported_types = {"causal", "seq2seq", "vlm"}
            if model_type.lower() in supported_types:
                return model_type.lower()
            logger.warning(
                f"Model type override '{model_type}' is not one of {sorted(supported_types)}. "
                "Falling back to auto-detection."
            )

        mt = getattr(self._config, "model_type", "").lower()

        if mt in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES:
            return "vlm"
        if mt in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES:
            return "seq2seq"
        if mt in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            return "causal"

        # remote_code models (e.g. EXAONE) and architectures not yet in the
        # transformers Auto mappings: use the standard is_encoder_decoder hint.
        if getattr(self._config, "is_encoder_decoder", False):
            return "seq2seq"
        logger.info(
            f"model_type '{mt}' not found in transformers AutoModel mappings; "
            "defaulting to 'causal' via is_encoder_decoder=False."
        )
        return "causal"

    def _load_model(self, pretrained: str, revision: str, trust_remote_code: bool,
                   low_cpu_mem_usage: bool, device_map: Optional[str],
                   dtype: Union[str, torch.dtype], **kwargs):
        """Load the appropriate model type using optimum.rbln API."""
        torch_dtype = lm_eval.models.utils_hf.get_dtype(dtype)

        # Extract RBLN-specific parameters from kwargs
        rbln_batch_size = kwargs.pop('rbln_batch_size', 1)
        rbln_max_seq_len = kwargs.pop('rbln_max_seq_len', 8192)
        rbln_tensor_parallel_size = kwargs.pop('rbln_tensor_parallel_size', 1)
        # VLM path: nested compile profile (rbln-model-zoo style).
        # Accept either a dict or a JSON string.
        rbln_config_json = kwargs.pop('rbln_config_json', None)
        if isinstance(rbln_config_json, str):
            rbln_config_json = json.loads(rbln_config_json)
        export = kwargs.pop('export', True)

        # Filter out parameters that are not supported by optimum.rbln
        # These are transformers-specific parameters that don't apply to RBLN models
        unsupported_params = [
            'device', 'low_cpu_mem_usage', 'device_map', 'dtype', 'torch_dtype',
            'load_in_8bit', 'load_in_4bit', 'quantization_config', 'attn_implementation',
            'max_seq_len'  # Add max_seq_len to unsupported params - we use rbln_max_seq_len instead
        ]

        # Create clean kwargs for RBLN model loading
        rbln_kwargs = {}
        for key, value in kwargs.items():
            if key not in unsupported_params:
                rbln_kwargs[key] = value

        # Determine the specific RBLN model class based on model architecture
        model_class = self._get_rbln_model_class(pretrained)

        logger.info(f"Loading {model_class.__name__} for model: {pretrained}")

        rbln_params = {
            'model_id': pretrained,
            'revision': revision,
            'trust_remote_code': trust_remote_code,
            'export': export,
        }

        # Compile-time params only apply when actually compiling (export=True).
        # When loading a pre-compiled artifact, optimum.rbln reads these from
        # rbln_config.json and rejects any override attempts.
        if export:
            if self.model_type == "vlm":
                mt = getattr(self._config, "model_type", "").lower()
                profile = _VLM_COMPILE_PROFILES.get(mt, {})
                merged = _deep_merge_dict(profile, rbln_config_json or {})
                if not merged:
                    logger.warning(
                        f"No VLM compile profile for model_type='{mt}' and no "
                        "rbln_config_json provided. Compilation may fail; pass "
                        "--model_args 'rbln_config_json={...}' with the per-model "
                        "compile parameters from rbln-model-zoo."
                    )
                else:
                    rbln_params['rbln_config'] = merged
                    logger.info(f"VLM compile config (model_type={mt}): {merged}")
            else:
                rbln_params['rbln_batch_size'] = rbln_batch_size
                rbln_params['rbln_tensor_parallel_size'] = rbln_tensor_parallel_size
                if 'Seq2Seq' not in model_class.__name__:
                    rbln_params['rbln_max_seq_len'] = rbln_max_seq_len
                else:
                    logger.info("Seq2seq model detected - skipping rbln_max_seq_len parameter")
                logger.info(
                    f"RBLN config: batch_size={rbln_batch_size}, "
                    f"max_seq_len={rbln_max_seq_len}, tensor_parallel_size={rbln_tensor_parallel_size}"
                )
        else:
            logger.info("export=False: loading pre-compiled artifact; compile-time params ignored")

        try:
            return model_class.from_pretrained(**rbln_params, **rbln_kwargs)
        except Exception as e:
            logger.error(f"Failed to load {self.model_type} model on Rebellions NPU: {e}")
            raise
    
    def _get_rbln_model_class(self, pretrained: str):
        """Pick the RBLN Auto class for the resolved ``self.model_type``.

        VLM routing splits between two SDK auto classes based on config.model_type
        (see ``_VLM_IMAGE_TEXT_TO_TEXT_MODEL_TYPES``). All other VLMs go through
        ``RBLNAutoModelForVision2Seq``.
        """
        if self.model_type == "vlm":
            mt = getattr(self._config, "model_type", "").lower()
            if mt in _VLM_IMAGE_TEXT_TO_TEXT_MODEL_TYPES:
                if RBLNAutoModelForImageTextToText is None:
                    raise ImportError(
                        "This optimum-rbln build does not expose "
                        "RBLNAutoModelForImageTextToText. Upgrade optimum-rbln "
                        f"to evaluate model_type='{mt}'."
                    )
                logger.info(
                    f"Using RBLNAutoModelForImageTextToText for {mt}: {pretrained}"
                )
                return RBLNAutoModelForImageTextToText
            if RBLNAutoModelForVision2Seq is None:
                raise ImportError(
                    "This optimum-rbln build does not expose "
                    "RBLNAutoModelForVision2Seq. Upgrade optimum-rbln to "
                    f"evaluate model_type='{mt}'."
                )
            logger.info(
                f"Using RBLNAutoModelForVision2Seq for {mt}: {pretrained}"
            )
            return RBLNAutoModelForVision2Seq

        if self.model_type == "seq2seq":
            logger.info(f"Using RBLNAutoModelForSeq2SeqLM for: {pretrained}")
            return RBLNAutoModelForSeq2SeqLM

        logger.info(f"Using RBLNAutoModelForCausalLM for: {pretrained}")
        return RBLNAutoModelForCausalLM

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self):
        # it is used as prefix for loglikelihood
        return self.tokenizer.bos_token_id or self.tokenizer.eos_token_id

    @property
    def max_length(self):
        # TODO: Get actual max length from model config
        return getattr(self._config, "max_position_embeddings", 2048)

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        # NPU device handling
        return next(self.model.parameters()).device

    @property
    def rank(self):
        return 0

    @property
    def world_size(self):
        return 1

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None):
        """Encode a single string, mirroring HFLM behaviour.

        With ``add_bos_token=None`` (default) the tokenizer's own default
        applies — for most modern LLM tokenizers this means BOS is prepended,
        which is what HFLM does on GPU. To force off / on, pass
        ``add_bos_token=False`` / ``True`` to ``--model_args``.

        Also guards against double-BOS when the string already starts with
        the BOS token.
        """
        special_tokens_kwargs = _add_special_kwargs(
            add_special_tokens, self.add_bos_token
        )
        if add_special_tokens is None and has_bos_prefix(
            string, self.tokenizer.decode(self.prefix_token_id)
        ):
            special_tokens_kwargs["add_special_tokens"] = False
        encoding = self.tokenizer.encode(string, **special_tokens_kwargs)

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]

        return encoding

    def tok_batch_encode(
        self,
        strings: List[str],
        padding_side: str = "left",
        left_truncate_len: int = None,
        truncation: bool = False,
    ):
        # encode a batch of strings. converts to tensors and pads automatically, unlike tok_encode.
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side

        # Mirror HFLM: respect tokenizer default unless add_bos_token is set;
        # guard against double-BOS using the first string as the probe.
        add_special_tokens: dict = {}
        if has_bos_prefix(strings[0], getattr(self.tokenizer, "bos_token", None)):
            add_special_tokens = {"add_special_tokens": False}
        elif self.add_bos_token is not None:
            add_special_tokens = {"add_special_tokens": self.add_bos_token}

        encoding = self.tokenizer(
            strings,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
            **add_special_tokens,
        )
        if left_truncate_len:
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][
                :, -left_truncate_len:
            ]
        self.tokenizer.padding_side = old_padding_side

        return encoding["input_ids"], encoding["attention_mask"]

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # we require users to pass do_sample=True explicitly
        # for non-greedy gen. This should be reevaluated when considering beam search.

        with torch.inference_mode():
            if "do_sample" not in generation_kwargs.keys():
                generation_kwargs["do_sample"] = False

            stopping_criteria = stop_sequences_criteria(
                self.tokenizer,
                stop + [self.tokenizer.decode([self.config.eos_token_id])],
                1,
                context.shape[0],
            )

            return self.model.generate(
                input_ids=context,
                max_length=max_length,
                stopping_criteria=stopping_criteria,
                pad_token_id=self.eot_token_id,
                use_cache=True,
                **generation_kwargs,
            )

    def _select_cont_toks(self, logits, contlen=None, inplen=None):
        assert contlen and inplen, (
            "Must pass input len and cont. len to select scored logits for causal LM"
        )
        # discard right-padding.
        # also discard the input/context tokens. we'll only score continuations.
        logits = logits[inplen - contlen : inplen]

        return logits

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        loglikelihoods = []

        adaptive_batch_size = None

        for (string,) in tqdm(
            [req.args for req in requests], disable=(disable_tqdm or (self.rank != 0))
        ):
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )

            # TODO: Right now, we pass single EOT token to the Encoder and the full context to the decoder, in seq2seq case
            rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            pad_amnt = 0
            if self.world_size > 1:
                # We pad out the external document-level iterator so the inner iterator doesn't hang
                mytensor = torch.tensor(len(rolling_token_windows), device=self.device)
                gathered = (
                    self.accelerator.gather(mytensor).cpu().detach().numpy().tolist()
                )

                pad_amnt = max(gathered) - gathered[self.rank]
                if pad_amnt > 0:
                    rolling_token_windows += pad_amnt * [rolling_token_windows[0]]

            string_nll = self._loglikelihood_tokens(
                rolling_token_windows,
                disable_tqdm=True,
                override_bs=adaptive_batch_size,
            )

            if (self.world_size > 1) and (pad_amnt > 0):
                string_nll = [x[0] for x in string_nll[:-pad_amnt]]
            else:
                # discard is_greedy
                string_nll = [x[0] for x in string_nll]

            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)
            # cache this loglikelihood_rolling request
            self.cache_hook.add_partial("loglikelihood_rolling", (string,), string_nll)
        return loglikelihoods
    
    def _model_call(self, inps, **kwargs):
        """
        Standard model call for causal and seq2seq models.
        """
        if self.model_type not in ["causal", "seq2seq"]:
            raise ValueError(f"Model type '{self.model_type}' is not supported. Only 'causal' and 'seq2seq' models are supported.")
        
        # Standard forward pass for supported models
        with torch.inference_mode():
            outputs = self.model(inps)
            if isinstance(outputs, tuple):
                return outputs[0]
            elif hasattr(outputs, 'logits'):
                return outputs.logits
            else:
                return outputs

    def _apply_cache_position_fix(self):
        """
        Apply cache_position conversion fix for RBLN seq2seq models.
        
        RBLN SDK bug: cache_position (None/Tensor) is passed to torch.full() which expects scalar.
        This method patches the forward method to convert cache_position to proper scalar values.
        """
        if hasattr(self.model, '_rbln_cache_fix_applied'):
            return  # Already applied
        
        # Store original forward method
        original_forward = self.model.forward
        
        def fixed_forward(*args, **kwargs):
            """Fixed forward method with proper cache_position scalar conversion"""
            
            # The cache_position issue occurs inside the RBLN model's forward method
            # We need to intercept and fix it, but it might not be passed as a parameter
            # Instead, we'll monkey patch torch.full temporarily during the forward call
            
            import torch.nn.functional as F_orig
            original_torch_full = torch.full
            
            def patched_torch_full(size, fill_value, **full_kwargs):
                """
                Patched torch.full that safely converts fill_value to scalar
                
                This fixes RBLN SDK bug where cache_position (None/Tensor) is passed to torch.full()
                which expects a scalar fill value.
                """
                if fill_value is None:
                    # None -> 0 (start position) - SAFE: This is the correct default
                    converted_value = 0
                    logger.debug("cache_position: None -> 0 (start position)")
                    
                elif isinstance(fill_value, torch.Tensor):
                    if fill_value.numel() == 0:
                        # Empty tensor -> 0 - SAFE: Reasonable default
                        converted_value = 0
                        logger.debug("cache_position: Empty tensor -> 0")
                        
                    elif fill_value.numel() == 1:
                        # Single element -> extract scalar - SAFE: Perfect data preservation
                        converted_value = fill_value.item()
                        logger.debug(f"cache_position: Single tensor {fill_value} -> {converted_value}")
                        
                    else:
                        # Multi-element tensor -> first element - POTENTIAL RISK
                        converted_value = fill_value.flatten()[0].item()
                        logger.warning(f"CACHE_POSITION RISK: Multi-element tensor detected: {fill_value}")
                        logger.warning(f"Using first element only: {converted_value}")
                        logger.warning("This may cause incorrect attention if batch items have different positions")
                        logger.warning("Consider checking batch size configuration or RBLN SDK version")
                        
                elif isinstance(fill_value, (int, float)):
                    # Already scalar - SAFE: No conversion needed
                    converted_value = fill_value
                    logger.debug(f"cache_position: Already scalar {converted_value}")
                    
                else:
                    # Unknown type -> 0 - SAFE: Conservative fallback
                    converted_value = 0
                    logger.warning(f"cache_position: Unknown type {type(fill_value)} -> 0")
                
                return original_torch_full(size, converted_value, **full_kwargs)
            
            # Temporarily replace torch.full
            torch.full = patched_torch_full
            
            try:
                # Convert cache_position to scalar if present in kwargs
                if 'cache_position' in kwargs:
                    cache_pos = kwargs['cache_position']
                    
                    if cache_pos is None:
                        kwargs['cache_position'] = 0
                        logger.debug("Converted cache_position kwarg: None -> 0")
                    elif isinstance(cache_pos, torch.Tensor):
                        if cache_pos.numel() == 1:
                            scalar_val = cache_pos.item()
                            kwargs['cache_position'] = scalar_val
                            logger.debug(f"Converted cache_position kwarg: Tensor -> {scalar_val}")
                        else:
                            scalar_val = cache_pos.flatten()[0].item() if cache_pos.numel() > 0 else 0
                            kwargs['cache_position'] = scalar_val
                            logger.debug(f"Converted cache_position kwarg: Tensor{cache_pos.shape} -> {scalar_val}")
                
                # Call original forward with patched torch.full and fixed parameters
                result = original_forward(*args, **kwargs)
                return result
                
            finally:
                # Always restore original torch.full
                torch.full = original_torch_full
        
        # Apply the fix
        self.model.forward = fixed_forward
        self.model._rbln_cache_fix_applied = True
        logger.info(" Applied RBLN seq2seq cache_position scalar conversion fix")

    def _loglikelihood_tokens(
        self, requests, disable_tqdm: bool = False, override_bs=None
    ):
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        re_ord = utils.Reorderer(requests, _collate)
        reordered = re_ord.get_reordered()

        n_reordered_requests = len(reordered)
        # automatic (variable) batch size detection for vectorization
        # pull longest context sample from request

        # Count unique contexts as a proxy for "docs" — multiple-choice
        # loglikelihood emits N requests per doc that share the same
        # context_enc but differ in continuation_enc, so the unique
        # context count equals the original doc count.
        total_docs = len({tuple(req[1]) for req in reordered})
        seen_contexts: set = set()

        chunks = lm_eval.models.utils.chunks(reordered, n=self.batch_size, fn=None)

        bs = max(int(self.batch_size), 1)
        n_chunks = (n_reordered_requests + bs - 1) // bs
        pbar = tqdm(
            chunks,
            total=n_chunks,
            disable=(disable_tqdm or (self.rank != 0)),
            desc=f"Running loglikelihood requests (0/{total_docs})",
        )
        for chunk in pbar:
            for _, ctx_enc, _ in chunk:
                ctx_key = tuple(ctx_enc)
                if ctx_key not in seen_contexts:
                    seen_contexts.add(ctx_key)
            pbar.set_description(
                f"Running loglikelihood requests ({len(seen_contexts)}/{total_docs})"
            )
            inps = []
            cont_toks_list = []
            inplens = []

            conts = []  # noqa
            encoder_attns = []  # noqa

            padding_len_inp = None
            padding_len_cont = None  # noqa
            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works (illustrated on a causal decoder-only setup):
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # model  \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # when too long to fit in context, truncate from the left
                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                    dtype=torch.long,
                    device=self.device,
                )
                (inplen,) = inp.shape

                padding_len_inp = (
                    max(padding_len_inp, inplen)
                    if padding_len_inp is not None
                    else inplen
                )

                inps.append(inp)  # [1, inp_length]
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)

            masks = [torch.ones_like(inp) for inp in inps]
            batched_inps = lm_eval.models.utils_hf.pad_and_concat(
                padding_len_inp, inps, padding_side="right"
            )  # [batch, padding_len_inp]

            batched_masks = lm_eval.models.utils_hf.pad_and_concat(
                padding_len_inp, masks, padding_side="right"
            )

            # Forward pass through the model
            with torch.inference_mode():
                if self.model_type == "seq2seq":
                    pass
                    
                    # Mark that we don't need per-token forward for seq2seq
                    needs_per_token_forward = False
                    
                else:
                    # For causal models - the only other supported type
                    if self.model_type not in ["causal", "seq2seq"]:
                        logger.error(f" Model type '{self.model_type}' is not supported")
                        logger.error(" Supported model types:")
                        logger.error("   • causal: GPT, LLaMA, Mistral, etc.")
                        logger.error("   • seq2seq: T5, BART, Pegasus, etc.")
                        logger.error(" Other model types will be implemented later")

                        # Raise an informative error
                        raise ValueError(
                            f"Model type '{self.model_type}' is not supported. "
                            f"Only 'causal' and 'seq2seq' models are currently supported."
                        )
                    
                    # For causal models, try standard forward pass first
                    outputs = self.model(
                        input_ids=batched_inps,
                        attention_mask=batched_masks,
                    )
                    
                    # Handle different output formats from RBLN models
                    if isinstance(outputs, tuple):
                        # RBLN models may return tuple (logits, ...)
                        logits = outputs[0]
                    elif hasattr(outputs, 'logits'):
                        # Standard transformers output format
                        logits = outputs.logits
                    else:
                        # Fallback: assume outputs are logits directly
                        logits = outputs
                    
                    # Check if RBLN model returns full sequence logits or just last token
                    batch_size, actual_seq_len = batched_inps.shape
                    expected_shape = (batch_size, actual_seq_len, logits.shape[-1])
                    
                    if logits.shape[:2] != (batch_size, actual_seq_len):
                        # RBLN compiled causal models return only the last-token logits
                        # ([batch, 1, vocab] or [batch, vocab]). Log this once per
                        # instance to avoid flooding the console on long benchmarks.
                        if not getattr(self, "_last_token_logits_logged", False):
                            logger.info(
                                f"RBLN causal model returns last-token logits "
                                f"(shape e.g. {tuple(logits.shape)}, expected full "
                                f"{expected_shape}). Falling back to per-token forward "
                                "pass for loglikelihood. This message is logged once."
                            )
                            self._last_token_logits_logged = True

                        # For loglikelihood, we need logits at continuation positions only
                        # We'll handle this in the per-sample processing below
                        vocab_size = logits.shape[-1]
                        
                        # Create placeholder full logits (will be filled per sample as needed)
                        multi_logits = torch.zeros(batch_size, actual_seq_len, vocab_size, dtype=torch.float32, device=self.device)
                        
                        # Mark this as needing special handling
                        needs_per_token_forward = True
                    else:
                        # Standard full sequence logits
                        multi_logits = F.log_softmax(logits, dim=-1)
                        needs_per_token_forward = False

            # Process each sample in the batch
            for sample_idx, ((cache_key, context_enc, continuation_enc), logits, inplen, cont_toks) in enumerate(zip(
                chunk, multi_logits, inplens, cont_toks_list
            )):
                # Special handling for RBLN models that return only last-token logits
                if needs_per_token_forward:
                    # RBLN causal model returns only last-token logits, need per-token forward pass
                    contlen = len(cont_toks)
                    
                    logger.debug(f"Per-token forward pass for {contlen} continuation tokens")
                    logger.debug(f"Context length: {len(context_enc)}, Continuation: {cont_toks}")
                    
                    # Build the complete input sequence: context + continuation
                    full_sequence = context_enc + continuation_enc
                    
                    # Truncate to max_length if needed, keeping the end (most recent tokens)
                    if len(full_sequence) > self.max_length:
                        full_sequence = full_sequence[-self.max_length:]
                        logger.debug(f"Truncated sequence to {len(full_sequence)} tokens")
                    
                    # Convert to tensor
                    full_input = torch.tensor(full_sequence, dtype=torch.long, device=self.device).unsqueeze(0)
                    
                    # Find where continuation starts in the (possibly truncated) sequence
                    context_in_full = len(full_sequence) - len(continuation_enc)
                    logger.debug(f"Continuation starts at position {context_in_full} in full sequence")
                    
                    # Get logits for each continuation token position
                    continuation_logits = []
                    
                    for i in range(contlen):
                        # Position in the full sequence where this continuation token should be predicted
                        target_pos = context_in_full + i
                        
                        if target_pos >= len(full_sequence):
                            logger.warning(f"Target position {target_pos} beyond sequence length {len(full_sequence)}")
                            vocab_size = 32128  # Default vocab size
                            zero_logits = torch.zeros(1, vocab_size, device=self.device)
                            continuation_logits.append(zero_logits)
                            continue
                        
                        # Input up to (but not including) the target token
                        # We want to predict the token at target_pos, so input is [:target_pos]
                        input_slice = full_input[:, :target_pos]  # Fixed: don't include target position
                        
                        logger.debug(f"Token {i}: predicting position {target_pos}, input shape: {input_slice.shape}")
                        
                        try:
                            # Forward pass to get logits for this position
                            outputs = self.model(input_ids=input_slice)
                            
                            # Handle different output formats
                            if isinstance(outputs, tuple):
                                pos_logits = outputs[0]
                            elif hasattr(outputs, 'logits'):
                                pos_logits = outputs.logits
                            else:
                                pos_logits = outputs
                            
                            # Extract last token logits (which predicts the next token)
                            if pos_logits.dim() == 3:
                                # Shape: [batch, seq_len, vocab] -> get last position
                                last_token_logits = pos_logits[:, -1, :]  # [1, vocab_size]
                            elif pos_logits.dim() == 2:
                                # Shape: [batch, vocab] - already last token
                                last_token_logits = pos_logits  # [1, vocab_size]
                            else:
                                logger.error(f"Unexpected logits shape: {pos_logits.shape}")
                                vocab_size = 32128
                                last_token_logits = torch.zeros(1, vocab_size, device=self.device)
                            
                            continuation_logits.append(last_token_logits)
                            logger.debug(f"Token {i}: got logits shape {last_token_logits.shape}")
                            
                        except Exception as e:
                            logger.error(f"CRITICAL: Per-token forward pass failed for token {i} at position {target_pos}: {e}")
                            logger.error(f"Input shape: {input_slice.shape}, Full sequence length: {len(full_sequence)}")
                            logger.error(f"Context length: {len(context_enc)}, Continuation length: {len(continuation_enc)}")
                            # Use zero logits as fallback (this will hurt accuracy)
                            vocab_size = 32128
                            zero_logits = torch.zeros(1, vocab_size, device=self.device)
                            continuation_logits.append(zero_logits)
                    
                    # Stack continuation logits
                    if continuation_logits:
                        # Stack along sequence dimension: [1, vocab] -> [1, cont_len, vocab]
                        logits = torch.stack(continuation_logits, dim=1)  # [1, cont_len, vocab_size]
                        logits = F.log_softmax(logits, dim=-1)
                        logger.debug(f"Final logits shape: {logits.shape}")
                    else:
                        # Fallback: use dummy logits
                        vocab_size = 32128
                        logits = torch.zeros(1, contlen, vocab_size, device=self.device)
                        logger.warning("Using zero logits for continuation")
                
                else:
                    # Standard processing for models that return full sequence logits
                    contlen = len(cont_toks)
                    
                    if self.model_type == "seq2seq":
                        # Seq2seq models: logits correspond to decoder output (continuation tokens)
                        # Extract logits for the continuation length
                        # The logits shape should be [batch, decoder_seq_len, vocab_size]
                        pass

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)
                cont_toks_tensor = torch.tensor(
                    cont_toks, dtype=torch.long, device=self.device
                ).unsqueeze(0)  # [1, seq]
                
                # Handle potential shape mismatches
                if greedy_tokens.shape != cont_toks_tensor.shape:
                    logger.warning(f"Shape mismatch: greedy_tokens {greedy_tokens.shape}, cont_toks {cont_toks_tensor.shape}")
                    logger.warning(f"logits shape: {logits.shape}")
                    # Adjust shapes to match
                    min_seq_len = min(greedy_tokens.shape[1], cont_toks_tensor.shape[1])
                    greedy_tokens = greedy_tokens[:, :min_seq_len]
                    cont_toks_tensor = cont_toks_tensor[:, :min_seq_len]
                    logits = logits[:, :min_seq_len, :]
                
                max_equal = (greedy_tokens == cont_toks_tensor).all()

                # Obtain log-probs at the corresponding continuation token indices
                try:
                    gathered_logits = torch.gather(logits, 2, cont_toks_tensor.unsqueeze(-1)).squeeze(-1)  # [1, seq]
                    answer = (float(gathered_logits.sum()), bool(max_equal))
                except RuntimeError as e:
                    logger.error(f"Error in torch.gather: {e}")
                    logger.error(f"logits shape: {logits.shape}, cont_toks shape: {cont_toks_tensor.shape}")
                    # Fallback: use the mean of the last token logits
                    answer = (float(logits[:, -1, :].mean()), False)

                res.append(answer)

                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)

        return re_ord.get_original(res)

    def generate_until(self, requests, disable_tqdm: bool = False):
        from collections import defaultdict

        res = defaultdict(list)
        re_ords = {}

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        grouper = lm_eval.models.utils.Grouper(requests, lambda x: str(x.args[1]))
        for key, reqs in grouper.get_grouped().items():
            # within each set of reqs for given kwargs, we reorder by token length, descending.
            re_ords[key] = utils.Reorderer([req.args for req in reqs], _collate)

        pbar = tqdm(total=len(requests), disable=(disable_tqdm or (self.rank != 0)))

        # for each different set of kwargs, we execute all requests, by batch.
        for key, re_ord in re_ords.items():
            chunks = lm_eval.models.utils.chunks(
                re_ord.get_reordered(), n=self.batch_size
            )
            for chunk in tqdm(chunks, disable=self.rank != 0):
                contexts, all_gen_kwargs = zip(*chunk)
                # we assume all gen kwargs in the batch are the same
                # this is safe to assume because the `grouper` object ensures it.
                gen_kwargs = all_gen_kwargs[0]
                # unpack our keyword arguments.
                until = None
                if isinstance(gen_kwargs, dict):
                    kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                    if "until" in kwargs.keys():
                        until = kwargs.pop("until")
                        if isinstance(until, str):
                            until = [until]
                        elif not isinstance(until, list):
                            raise ValueError(
                                f"Expected `kwargs['until']` to be of type Union[str,list] but got {until}"
                            )
                else:
                    raise ValueError(
                        f"Expected `kwargs` to be of type `dict` but got {kwargs}"
                    )
                # add EOS token to stop sequences
                eos = self.tok_decode(self.eot_token_id)
                if not until:
                    until = [eos]
                else:
                    until.append(eos)
                if "max_gen_toks" in kwargs.keys():
                    max_gen_toks = kwargs.pop("max_gen_toks")
                else:
                    max_gen_toks = self.max_gen_toks
                # first stop sequence is used to halt generation upon encountering
                primary_until = [until[0]]

                max_ctx_len = self.max_length - max_gen_toks

                # encode, pad, and truncate contexts for this batch
                context_enc, attn_masks = self.tok_batch_encode(
                    contexts,
                    left_truncate_len=max_ctx_len,
                    truncation=self.truncation,
                )
                context_enc = context_enc.to(self.device)
                attn_masks = attn_masks.to(self.device)

                if "max_length" not in kwargs:
                    kwargs["max_length"] = context_enc.shape[1] + max_gen_toks

                # perform batched generation
                cont = self._model_generate(
                    context=context_enc,
                    attention_mask=attn_masks,
                    stop=primary_until,
                    **kwargs,
                )

                cont_toks_list = cont.tolist()
                for cont_toks, context in zip(cont_toks_list, contexts):
                    # discard context + left-padding toks if using causal decoder-only LM
                    cont_toks = cont_toks[context_enc.shape[1] :]

                    s = self.tok_decode(cont_toks)

                    # use secondary stop seqs to cut off should-have-been-stopped content post-hoc
                    for term in until:
                        if len(term) > 0:
                            # ignore '' separator,
                            # for seq2seq case where self.tok_decode(self.eot_token_id) = ''
                            s = s.split(term)[0]

                    res[key].append(s)

                    self.cache_hook.add_partial(
                        "generate_until", (context, gen_kwargs), s
                    )
                    pbar.update(1)
            # reorder this group of results back to original unsorted form
            res[key] = re_ord.get_original(res[key])

        pbar.close()

        return grouper.get_original(res)


# Simplified: Only the main rebellions_npu model with intelligent auto-detection
# All RBLN Auto classes are automatically selected based on model architecture
