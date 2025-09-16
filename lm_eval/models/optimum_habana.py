import logging
from importlib.util import find_spec
import os
import copy
from typing import Any
import torch
import torch.nn.functional as F

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils import get_dtype

from transformers import AutoModelForCausalLM
eval_logger = logging.getLogger(__name__)

@register_model("habana")
class HabanaLM(HFLM):
    """
    using the HuggingFace transformers + optimum-habana backend, can run on Intel Gaudi (HPU)
    """

    def __init__(
        self,
        **kwargs,
    ) -> None:

        """
        Intel Gaudi (HPU) extra --model_args arguments are:

        buckets: Optional[int] = [16, 32, 64, 128, 189, 284, 384],
        use_kv_cache: Optional[bool] = False,
        use_hpu_graphs: Optional[bool] = False,
        trim_logits: Optional[bool] = False,
        attn_softmax_bf16: Optional[bool] = False,
        bucket_internal: Optional[bool] = True,
        limit_hpu_graphs: Optional[bool] = False,
        clear_hpu_graphs_cache: Optional[bool] = False,
        show_graphs_count: Optional[bool] = False,
        reuse_cache: Optional[bool] = False,
        reduce_recompile: Optional[bool] = False,
        use_flex_attention: Optional[bool] = False,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        flash_attention_causal_mask: Optional[bool] = False,
        flash_attention_fast_softmax: Optional[bool] = False,
        torch_compile: Optional[bool] = True,
        sdp_on_bf16: Optional[bool] = False,
        regional_compile: Optional[bool] = True,
        cache_size_limit: Optional[int] = None,
        dynamo_specialize_float: Optional[bool] = True,
        dynamo_allow_unspec_int_on_nn_module: Optional[bool] = True,
        attn_batch_split: Optional[int] = 1,

        Default execution mode is Eager, if you prefer Lazy, add PT_HPU_LAZY_MODE=1 before lm_eval
        """

        if "backend" in kwargs:
            # currently only supports causal models
            assert kwargs["backend"] == "causal", (
                "Currently, only AutoModelForCausalLM is supported."
            )
        
        self.buckets = kwargs.pop("buckets", [16, 32, 64, 128])
        if isinstance(self.buckets, (int, str)):
            self.buckets = [self.buckets]
        self.buckets = [int(b) for b in self.buckets]

        if os.getenv("PT_HPU_LAZY_MODE", "0") == "0":
            self.lazy_mode = False
        else:
            self.lazy_mode = True

        super().__init__(
            backend=kwargs.pop("backend", "causal"),
            **kwargs,
        )
    def warm_up(self) -> None:
        for bucket_size in reversed(self.buckets):
            inps = torch.ones((self._batch_size, bucket_size), dtype=torch.int64)
            self._model_call(inps)
    
    def find_bucket(self, length: int, key=lambda b, length: b >= length) -> int:
        for b in self.buckets:
            if key(b, length):
                return b
        new_bucket = length
        self.buckets.append(new_bucket)
        self.buckets.sort()
        return new_bucket

    def _model_call(self, inps: torch.Tensor) -> torch.Tensor:
        bs, seq_length = inps.shape
        padding_length = 0
        if self.options.static_shapes:
            bucket_length = self.find_bucket(seq_length)
            if self.options.use_cache and self.options.reuse_cache:
                self._model.allocate_kv_cache(bs, bucket_length + 1, bucket_length)
            padding_length = bucket_length - seq_length
            inps = F.pad(inps, (0, padding_length), value=self._model.config.pad_token_id)
        logits = self._model(inps, **self.model_inputs)["logits"]

        if self.options.static_shapes and padding_length > 0:
            logits = logits[:, :-padding_length, :]
        logits = logits.to(torch.float32)

        return logits

    def setup_generation_config_gaudi(self,  **kwargs):
        # Add to the model config Intel Gaudi specific args

        generation_config = copy.deepcopy(self.config)
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
        #ToDo: Those values are not good for all models
        generation_config.clear_hpu_graphs_cache = kwargs.pop("clear_hpu_graphs_cache", True)
        generation_config.use_flex_attention = kwargs.pop("use_flex_attention", True)
        generation_config.use_flash_attention = kwargs.pop("use_flash_attention", True)
        generation_config.flash_attention_recompute = kwargs.pop("flash_attention_recompute", True)
        generation_config.flash_attention_causal_mask = kwargs.pop("flash_attention_causal_mask", True)
        generation_config.flash_attention_fast_softmax = kwargs.pop("flash_attention_fast_softmax", True)
        generation_config.reuse_cache = kwargs.pop("reuse_cache", True)

        return generation_config, kwargs
    
    def get_torch_compiled_model(self):
        #torch._dynamo.config.cache_size_limit = 64

        compile_kwargs = {
            "backend": "hpu_backend",
            "options": {"force_static_compile": False, "keep_input_mutations": True},
        }
        # for gpt_bigcode, mpt, bloom, gpt2 model_type
        if hasattr(self._model, "transformer"):
            self._model.transformer = torch.compile(self._model.transformer, **compile_kwargs)
        # for gpt_neox
        elif hasattr(self._model, "gpt_neox"):
            self._model.gpt_neox = torch.compile(self._model.gpt_neox, **compile_kwargs)
        # for llama, mistral, mixtral, qwen2
        elif hasattr(self._model, "model"):
            self._model.model = torch.compile(self._model.model, **compile_kwargs)
        else:
            self._model = torch.compile(self._model, **compile_kwargs)
        return self._model

    def _create_model(
        self,
        pretrained: str,
        revision="main",
        dtype="auto",
        trust_remote_code=False,        
	    parallelize=False,
        gpus=None,
        max_memory_per_gpu=None,
        max_cpu_memory=None,
        offload_folder="./offload",        
	    peft=None,
        delta=None,
        autogptq=False,
        gptqmodel=False,
        **kwargs,
    ) -> None:
        if not find_spec("optimum"):
            raise ModuleNotFoundError(
                "package `optimum-habana` is not installed. Please install it via `pip install optimum[habana]`"
            )
        else:
            from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
            adapt_transformers_to_gaudi()

        # Add Intel Gaudi specific options
        self.options, kwargs = self.setup_generation_config_gaudi(**kwargs)
        
        self.model_inputs = {"use_cache": self.options.use_cache,
                "reuse_cache": self.options.reuse_cache,
                "attn_softmax_bf16": self.options.attn_softmax_bf16,
                "use_flash_attention": self.options.use_flash_attention,
                "flash_attention_recompute": self.options.flash_attention_recompute,
                "flash_attention_causal_mask": self.options.flash_attention_causal_mask,
                "flash_attention_fast_softmax": self.options.flash_attention_fast_softmax}
        self.model_inputs.update({"use_flex_attention": self.options.use_flex_attention})

        model_kwargs = kwargs if kwargs else {}
        model_kwargs.update(
            self._get_accelerate_args(
                parallelize=parallelize,
                device_map=kwargs.get("device_map", None),
                max_memory_per_gpu=max_memory_per_gpu,
                max_cpu_memory=max_cpu_memory,
                offload_folder=offload_folder,
                gpus=gpus,
            )
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            revision=revision,
            torch_dtype=torch.bfloat16,
            trust_remote_code=trust_remote_code,
            **model_kwargs,
            )
        
        if self.lazy_mode:
            from habana_frameworks.torch.hpu import wrap_in_hpu_graph
            self._model = wrap_in_hpu_graph(self._model)
        else:
            self._model = self.get_torch_compiled_model()
    
    def _model_generate(
        self,
        context,
        max_length: int,
        stop: list[str],
        **generation_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """
        Patched method
        source: https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.9.1/lm_eval/models/huggingface.py#L951
        """
        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample")
        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        # build stopping criteria
        stopping_criteria = stop_sequences_criteria(self.tokenizer, stop, context.shape[1], context.shape[0])
        # to avoid graph recompilation
        if self.options.static_shapes:
            self.options.bucket_internal = True
            bucket_length = self.find_bucket(context.shape[1])
            padding_length = bucket_length - context.shape[1]
            max_gen_toks = max_length - context.shape[1]
            if padding_length > 0 and self.hpu_graphs:
                # Static shapes require right-padding (left-padding due to batch encoding is performed at tok_batch_encode level)
                # See https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.9.1/lm_eval/models/huggingface.py#L869
                context = F.pad(context, (0, padding_length), value=self.tokenizer.pad_token_id)
                generation_kwargs["attention_mask"] = F.pad(
                    generation_kwargs["attention_mask"], (0, padding_length), value=0
                )
        # move context & attention_mask to hpu
        context = context.to("hpu")
        generation_kwargs["attention_mask"] = generation_kwargs["attention_mask"].to("hpu")
        with torch.autocast(
            device_type="hpu",
            dtype=self.mixed_precision_dtype,
            enabled=self.mixed_precision_dtype is not None,
        ):
            return self.model.generate(
                input_ids=context,
                max_new_tokens=max_gen_toks,
                generation_config=self.options,
                stopping_criteria=stopping_criteria,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                hpu_graphs=self.hpu_graphs,
                lazy_mode=self.use_lazy_mode,
                **generation_kwargs,
            )
