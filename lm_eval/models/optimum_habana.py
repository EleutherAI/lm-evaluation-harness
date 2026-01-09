import logging
import os
from importlib.util import find_spec
from typing import Any

import torch
import torch.nn.functional as F

from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


eval_logger = logging.getLogger(__name__)


@register_model("habana")
class HabanaLM(HFLM):
    """
    HuggingFace transformers + optimum-habana backend, run on Intel Gaudi (HPU).

    Extra model_args:
        buckets: Optional[int] = [16, 32, 64, 128, 189, 284, 384],
        use_kv_cache: Optional[bool] = True,
        trim_logits: Optional[bool] = False,
        attn_softmax_bf16: Optional[bool] = True,
        bucket_internal: Optional[bool] = True,
        limit_hpu_graphs: Optional[bool] = True,
        clear_hpu_graphs_cache: Optional[bool] = True,
        reuse_cache: Optional[bool] = True,
        reduce_recompile: Optional[bool] = False,
        use_flex_attention: Optional[bool] = True,
        use_flash_attention: Optional[bool] = True,
        flash_attention_recompute: Optional[bool] = True,
        flash_attention_causal_mask: Optional[bool] = True,
        flash_attention_fast_softmax: Optional[bool] = True,
        sdp_on_bf16: Optional[bool] = False,
        attn_batch_split: Optional[int] = 1,
        ignore_eos: Optional[bool] = False

    Default execution mode is Eager, if you prefer Lazy, add PT_HPU_LAZY_MODE=1 before lm_eval
    """

    def __init__(self, **kwargs) -> None:
        # Validate backend
        if "backend" in kwargs:
            # currently only supports causal models
            assert kwargs["backend"] == "causal", (
                "Currently, only AutoModelForCausalLM is supported."
            )
        # Buckets
        self.buckets = kwargs.pop("buckets", [16, 32, 64, 128, 189, 284, 384])
        if isinstance(self.buckets, (int, str)):
            self.buckets = [self.buckets]
        self.buckets = [int(b) for b in self.buckets]
        self.lazy_mode = os.getenv("PT_HPU_LAZY_MODE", "0") != "0"
        self.options, kwargs = self.setup_generation_config_gaudi(**kwargs)
        super().__init__(backend=kwargs.pop("backend", "causal"), **kwargs)

    @property
    def max_length(self) -> int:
        # Better suits loglikelihood
        return self._max_length if self._max_length else self.buckets[-1]

    @max_length.setter
    def max_length(self, value: int) -> None:
        self._max_length = value

    def find_bucket(self, length: int, key=lambda b, length: b >= length) -> int:
        """
        Find the smallest bucket >= length, or add a new one.
        """
        for b in self.buckets:
            if key(b, length):
                return b
        new_bucket = length
        self.buckets.append(new_bucket)
        self.buckets.sort()
        eval_logger.info(
            f"Added new bucket: {new_bucket}. Buckets are now: {self.buckets}"
        )
        return new_bucket

    def _model_call(self, inps: torch.Tensor) -> torch.Tensor:
        """
        Calls the model with input tensor, handling static shape padding for HPU.
        """
        bs, seq_length = inps.shape
        padding_length = 0
        # Add padding at bucketing length
        if self.options.static_shapes:
            bucket_length = self.find_bucket(seq_length)
            if self.options.use_cache and self.options.reuse_cache:
                self._model.allocate_kv_cache(bs, bucket_length + 1, bucket_length)
            padding_length = bucket_length - seq_length
            pad_token_id = getattr(self._model.config, "pad_token_id", 0)
            inps = F.pad(inps, (0, padding_length), value=pad_token_id)
            eval_logger.debug(
                f"Padded input from {seq_length} to {bucket_length} (pad={padding_length})"
            )
        logits = super()._model_call(inps)

        if self.options.static_shapes and padding_length > 0:
            logits = logits[:, :-padding_length, :]
        return logits

    def setup_generation_config_gaudi(self, **kwargs):
        """
        Add to the model config Intel Gaudi specific args.
        """
        if not find_spec("optimum"):
            raise ModuleNotFoundError(
                "package `optimum-habana` is not installed. Please install it via `pip install optimum[habana]`"
            )
        else:
            from optimum.habana.transformers.generation import GaudiGenerationConfig

            generation_config = GaudiGenerationConfig()  # copy.deepcopy(self.config)
            generation_config.use_cache = kwargs.pop("use_kv_cache", True)
            generation_config.static_shapes = True
            generation_config.bucket_size = self.buckets
            generation_config.bucket_internal = kwargs.pop("bucket_internal", True)
            generation_config.trim_logits = kwargs.pop("trim_logits", False)
            generation_config.attn_softmax_bf16 = kwargs.pop("attn_softmax_bf16", True)
            generation_config.reduce_recompile = kwargs.pop("reduce_recompile", False)
            if generation_config.reduce_recompile:
                assert generation_config.bucket_size > 0
            generation_config.valid_sequence_lengths = None
            generation_config.attn_batch_split = kwargs.pop("attn_batch_split", 1)
            generation_config.limit_hpu_graphs = kwargs.pop("limit_hpu_graphs", True)
            generation_config.sdp_on_bf16 = kwargs.pop("sdp_on_bf16", False)
            generation_config.clear_hpu_graphs_cache = kwargs.pop(
                "clear_hpu_graphs_cache", True
            )
            generation_config.use_flex_attention = kwargs.pop(
                "use_flex_attention", True
            )
            generation_config.use_flash_attention = kwargs.pop(
                "use_flash_attention", True
            )
            generation_config.flash_attention_recompute = kwargs.pop(
                "flash_attention_recompute", True
            )
            generation_config.flash_attention_causal_mask = kwargs.pop(
                "flash_attention_causal_mask", True
            )
            generation_config.flash_attention_fast_softmax = kwargs.pop(
                "flash_attention_fast_softmax", True
            )
            generation_config.reuse_cache = kwargs.pop("reuse_cache", True)
            generation_config.ignore_eos = kwargs.pop("ignore_eos", False)
            return generation_config, kwargs

    def _create_model(self, *args, **kwargs) -> None:
        """
        Create and wrap the model for HPU, calling parent logic and then applying Gaudi specifics.
        """
        from optimum.habana.transformers.modeling_utils import (
            adapt_transformers_to_gaudi,
        )

        adapt_transformers_to_gaudi()
        super()._create_model(*args, **kwargs)

        if self.lazy_mode:
            from habana_frameworks.torch.hpu import wrap_in_hpu_graph

            self._model = wrap_in_hpu_graph(self._model)
            eval_logger.info("Model wrapped in HPU graph.")

    def generate_until(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[str]:
        """
        Override to change only max_length property
        """
        loglikelyhood_max_length = self.max_length
        self.max_length = super().max_length
        res = super().generate_until(requests, disable_tqdm)
        self.max_length = loglikelyhood_max_length
        return res

    def _model_generate(
        self,
        context: torch.Tensor,
        max_length: int,
        stop: list[str],
        **generation_kwargs: Any,
    ) -> torch.Tensor:
        """
        Generate tokens using the model, handling static shape padding and HPU specifics.
        """
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample")
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False
        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        if self.options.static_shapes:
            self.options.bucket_internal = True
            bucket_length = self.find_bucket(context.shape[1])
            padding_length = bucket_length - context.shape[1]
            max_gen_toks = max_length - context.shape[1]
            if padding_length > 0 and getattr(self, "lazy_mode", False):
                context = F.pad(
                    context, (0, padding_length), value=self.tokenizer.pad_token_id
                )
                generation_kwargs["attention_mask"] = F.pad(
                    generation_kwargs["attention_mask"], (0, padding_length), value=0
                )
        context = context.to(self.device)
        generation_kwargs["attention_mask"] = generation_kwargs["attention_mask"].to(
            self.device
        )
        with torch.autocast(
            device_type=str(self.device),
            dtype=self.mixed_precision_dtype,
            enabled=self.mixed_precision_dtype is not None,
        ):
            return self.model.generate(
                input_ids=context,
                max_new_tokens=max_gen_toks,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                hpu_graphs=getattr(self, "lazy_mode", False),  # To Simplify
                lazy_mode=getattr(self, "lazy_mode", False),
                **generation_kwargs,
            )
