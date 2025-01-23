from typing import Optional, Union

import torch

import lm_eval.models.utils
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


@register_model("rwkv")
class MambaLMWrapper(HFLM):
    def __init__(
        self,
        pretrained,
        # To use the HF compatible variant
        is_hf: bool = False,
        **kwargs,
    ) -> None:
        if "backend" in kwargs:
            assert kwargs["backend"] == "causal"
        self.is_hf = is_hf or (True if pretrained.endswith("hf") else False)
        assert kwargs["tokenizer"] is not None, "`tokenizer` is required"
        self.tokenizer = kwargs["tokenizer"]
        self.pretrained = pretrained
        super().__init__(
            pretrained=pretrained,
            # set appropriate defaults for tokenizer, max length, etc
            backend=kwargs.pop("backend", "causal"),
            tokenizer=self.tokenizer,
            max_length=kwargs.pop("max_length", 4096),
            **kwargs,
        )

    def _get_config(
        self,
        pretrained: str,
        **kwargs,
    ) -> None:
        if self.is_hf:
            super()._get_config(pretrained, **kwargs)
        else:
            self._config = {}

    def _create_model(
        self,
        pretrained: str,
        dtype: Optional[Union[str, torch.dtype]] = "fp16",
        **kwargs,
    ) -> None:
        if self.is_hf:
            super()._create_model(pretrained, dtype=dtype, **kwargs)
        else:
            try:
                from rwkv.model import RWKV
            except ModuleNotFoundError as exception:
                raise type(exception)(
                    "install rwkv package (pip install rwkv)",
                )

            import os

            os.environ["RWKV_JIT_ON"] = "1"
            os.environ["RWKV_CUDA_ON"] = "1"
            os.environ["RWKV_V7_ON"] = "1"

            self._model = RWKV(model=self.pretrained, strategy=f"cuda {dtype}")

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        remove_arg = (
            ["attention_mask"] if self.is_hf else ["do_sample", "attention_mask"]
        )
        for key in remove_arg:
            if key in generation_kwargs:
                generation_kwargs.pop(key)

        all_outputs = []
        if not self.is_hf:
            CHUNK_SIZE = 4096
            prefill_ids, next_token = context[:-1], context[-1]
            state = None
            for i in range(0, len(prefill_ids), CHUNK_SIZE):
                prefill_token = prefill_ids[i : i + CHUNK_SIZE]
                _, state = self.model(prefill_token, state)

            gen_length = context.shape[1] - max_length
            for i in range(gen_length):
                logits, state = self.model([next_token], state)
                next_token = torch.argmax(logits, dim=-1)
                all_outputs.append(next_token)

            return torch.cat(all_outputs)
        else:
            stopping_criteria = lm_eval.models.utils.stop_sequences_criteria(
                self.tokenizer,
                stop,
                context.shape[1],
                context.shape[0],
            )

            generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
            do_sample = generation_kwargs.get("do_sample", None)

            # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
            if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
                generation_kwargs["do_sample"] = do_sample = False
            if do_sample is False and generation_kwargs.get("temperature") == 0.0:
                generation_kwargs.pop("temperature")

            return self.model.generate(
                input_ids=context,
                max_length=max_length,
                stopping_criteria=stopping_criteria,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                **generation_kwargs,
            )
