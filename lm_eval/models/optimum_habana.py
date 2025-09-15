import logging
from importlib.util import find_spec
import os
import copy
import torch

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
        
        self.torch_compile = kwargs.pop("torch_compile", True)
        if os.getenv("PT_HPU_LAZY_MODE", "0") == "0":
            self.lazy_mode = False
        else:
            self.lazy_mode = True
            self.torch_compile = False

        super().__init__(
            backend=kwargs.pop("backend", "causal"),
            **kwargs,
        )

    def setup_generation_config_gaudi(self,  **kwargs):
        # Add to the model config Intel Gaudi specific args

        generation_config = {}
        generation_config["use_cache"] = kwargs.pop("use_kv_cache", True)
        generation_config["static_shapes"] = True
        generation_config["bucket_size"] = kwargs.pop("buckets", [16, 32, 64, 128, 189, 284, 384])
        generation_config["bucket_internal"] = kwargs.pop("bucket_internal", True)
        generation_config["trim_logits"] = kwargs.pop("trim_logits", False)
        generation_config["attn_softmax_bf16"] = kwargs.pop("attn_softmax_bf16", False)
        generation_config["reduce_recompile"] = kwargs.pop("reduce_recompile", False)
        if generation_config["reduce_recompile"]:
            assert generation_config["bucket_size"] > 0
        generation_config["valid_sequence_lengths"] = None
        generation_config["attn_batch_split"] = kwargs.pop("attn_batch_split", 1)
        if os.getenv("PT_HPU_LAZY_MODE", "0") == "1":
            generation_config["limit_hpu_graphs"] = kwargs.pop("limit_hpu_graphs", True)
            generation_config["clear_hpu_graphs_cache"] = kwargs.pop("clear_hpu_graphs_cache", True)
            generation_config["use_flex_attention"] = kwargs.pop("use_flex_attention", True)
            generation_config["use_flash_attention"] = kwargs.pop("use_flash_attention", True)
            generation_config["flash_attention_recompute"] = kwargs.pop("flash_attention_recompute", True)
            generation_config["flash_attention_causal_mask"] = kwargs.pop("flash_attention_causal_mask", True)
            generation_config["flash_attention_fast_softmax"] = kwargs.pop("flash_attention_fast_softmax", True)
            generation_config["reuse_cache"] = kwargs.pop("reuse_cache", True)
        else:
            generation_config["limit_hpu_graphs"] = kwargs.pop("limit_hpu_graphs", False)
            generation_config["clear_hpu_graphs_cache"] = kwargs.pop("clear_hpu_graphs_cache", False)
            generation_config["use_flex_attention"] = kwargs.pop("use_flex_attention", False)
            generation_config["use_flash_attention"] = kwargs.pop("use_flash_attention", False)
            generation_config["flash_attention_recompute"] = kwargs.pop("flash_attention_recompute", False)
            generation_config["flash_attention_causal_mask"] = kwargs.pop("flash_attention_causal_mask", False)
            generation_config["flash_attention_fast_softmax"] = kwargs.pop("flash_attention_fast_softmax", False)
            generation_config["reuse_cache"] = kwargs.pop("reuse_cache", False)

        return generation_config, kwargs

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
        self.generation_config, kwargs = self.setup_generation_config_gaudi(**kwargs)

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
            torch_dtype=get_dtype(dtype),
            trust_remote_code=trust_remote_code,
            **model_kwargs,
            )
        
        if self.lazy_mode:
            from habana_frameworks.torch.hpu import wrap_in_hpu_graph
            self._model = wrap_in_hpu_graph(self._model)

