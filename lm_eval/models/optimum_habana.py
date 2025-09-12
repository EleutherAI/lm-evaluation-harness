import logging
from importlib.util import find_spec
import os
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
        if "backend" in kwargs:
            # currently only supports causal models
            assert kwargs["backend"] == "causal", (
                "Currently, only AutoModelForCausalLM is supported."
            )

        if os.getenv("PT_HPU_LAZY_MODE", "0") == "0":
            self.lazy_mode = False
            self.hpu_graphs = False
        else:
            self.lazy_mode = True
            self.hpu_graphs = True

        super().__init__(
            backend=kwargs.pop("backend", "causal"),
            **kwargs,
        )

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

