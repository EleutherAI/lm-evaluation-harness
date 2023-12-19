import torch

import transformers
from transformers import AutoTokenizer

try:
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    from mamba_ssm.utils.hf import load_config_hf
except ModuleNotFoundError:
    pass

from lm_eval import utils
from lm_eval.api.model import LM
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model

from typing import Literal, Optional, Union


@register_model("mamba_ssm")
class MambaLMWrapper(HFLM):
    def __init__(
        self,
        pretrained="state-spaces/mamba-130m",
        # mamba supports most default HFLM kwargs.
        # however, it does not currently support advanced from_pretrained() kwargs
        # `parallelize=True`, PEFT, autoGPTQ,
        # or any sub-configurations of these advanced args.
        **kwargs,
    ) -> None:

        if "backend" in kwargs:
            # mamba currently only supports causal models
            assert kwargs["backend"] == "causal"

        super().__init__(
            pretrained=pretrained,
            # set appropriate defaults for tokenizer, max length, etc
            backend=kwargs.get("backend", "causal"),
            tokenizer=kwargs.get("tokenizer", "EleutherAI/gpt-neox-20b"),
            max_length=kwargs.get("max_length", 2048),
            **kwargs,
        )

    def _get_config(
        self,
        pretrained: str,
        **kwargs,
    ) -> None:

        try:
            from mamba_ssm.utils.hf import load_config_hf  # noqa: F811
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'mamba_ssm' LM type, but package `mamba_ssm` is not installed. \
please install mamba via `pip install lm-eval[mamba]` or `pip install -e .[mamba]`",
            )

        self._config = load_config_hf(pretrained)

    def _create_model(
        self,
        pretrained: str,
        dtype: Optional[Union[str, torch.dtype]] = "float16",
        # no `parallelize=True` options
        # no PEFT and quantization options
        # Mamba does not support arbitrary HF from_pretrained() args
        **kwargs,
    ) -> None:

        try:
            from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel  # noqa: F811
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'mamba_ssm' LM type, but package `mamba_ssm` is not installed. \
please install mamba via `pip install lm-eval[mamba]` or `pip install -e .[mamba]`",
            )

        self._model = MambaLMHeadModel.from_pretrained(
            pretrained,
            device=self._device,
            dtype=torch.float16 if dtype == "auto" else utils.get_dtype(dtype),
        )

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        for key in ("do_sample", "attention_mask"):
            if key in generation_kwargs:
                generation_kwargs.pop(key)

        # mamba's custom GenerationMixin currently does not support
        # passing stopping criteria.
        # for the time being, we simply generate to max length,
        # then truncate (equivalent result)
        # -- this should be revisited to speed up generation
        # stopping_criteria = stop_sequences_criteria(
        #     self.tokenizer, stop, 1, context.shape[0]
        # )

        return self.model.generate(
            input_ids=context,
            max_length=max_length,
            # stopping_criteria=stopping_criteria,
            # pad_token_id=self.tokenizer.pad_token_id,
            # use_cache=True,
            **generation_kwargs,
        )
