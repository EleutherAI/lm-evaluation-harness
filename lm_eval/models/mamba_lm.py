from typing import Optional, Union

import torch

import lm_eval.models.utils
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


@register_model("mamba_ssm")
class MambaLMWrapper(HFLM):
    def __init__(
        self,
        pretrained="state-spaces/mamba-130m",
        **kwargs,
    ) -> None:
        """
        Mamba (via the `mamba_ssm` package) supports the following args:
        ```
        d_model: int,
        n_layer: int,
        vocab_size: int,
        initializer_cfg=None,
        pad_vocab_size_multiple: int = 1,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        ```

        See https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L175 for more info.
        The above can all be passed via `--model_args` or to this __init__() directly
        but we recommend placing many of these within the config.json file uploaded alongside your
        Mamba model to the HF Hub instead.
        All other HuggingFace from_pretrained() kwargs
        such as those related to
        `parallelize=True`, PEFT, autoGPTQ,
        or any sub-configurations of these advanced args,
        are unsupported by the `mamba_ssm` package.

        The HFLM arguments

        `backend`, `tokenizer`, `truncation`, `max_length`,
        `device`, `dtype`, `batch_size`, `max_batch_size`, `trust_remote_code`, `use_fast_tokenizer`

        Are all supported by Mamba where they do not conflict
        with Mamba-specific restrictions such as causal LMs only.
        """

        if "backend" in kwargs:
            # mamba currently only supports causal models
            assert kwargs["backend"] == "causal"

        super().__init__(
            pretrained=pretrained,
            # set appropriate defaults for tokenizer, max length, etc
            backend=kwargs.pop("backend", "causal"),
            tokenizer=kwargs.pop("tokenizer", "EleutherAI/gpt-neox-20b"),
            max_length=kwargs.pop("max_length", 2048),
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
            dtype=torch.float16
            if dtype == "auto"
            else lm_eval.models.utils.get_dtype(dtype),
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
