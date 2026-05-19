"""
RBLN NPU multimodal (Vision-Language) model integration for lm_eval.

This module adds a ``RBLNVLM`` class registered as ``rbln-vlm``. It mirrors the
generate-only multimodal pattern used by ``HFMultimodalLM`` (see
``lm_eval/models/hf_vlms.py``) but routes model loading through the existing
text-only ``RBLNLM`` (``optimum.rbln`` Auto classes) and uses the per-model
VLM compile profiles defined in ``optimum_rbln._VLM_COMPILE_PROFILES``.

Currently supports:
- ``generate_until`` requests (used by MMMU, ChartQA).
- Falls back to the inherited ``RBLNLM`` text path when a request has no
  ``aux_arguments["visual"]`` payload.

Not yet supported:
- ``loglikelihood`` / ``loglikelihood_rolling`` (raises NotImplementedError,
  matching ``HFMultimodalLM`` behavior).
"""

import copy
import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import transformers
from tqdm import tqdm
from transformers import BatchEncoding

from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.models.optimum_rbln import RBLNLM
from lm_eval.models.utils import (
    Collator,
    flatten_image_list,
    handle_stop_sequences,
    replace_placeholders,
    resize_image,
)
from lm_eval.models.utils_hf import stop_sequences_criteria


DEFAULT_IMAGE_PLACEHOLDER = "<image>"

logger = logging.getLogger(__name__)


@register_model("rbln-vlm")
class RBLNVLM(RBLNLM):
    """RBLN NPU runner for vision-language models (e.g. LLaVA, Qwen2.5-VL, Gemma3)."""

    MULTIMODAL = True

    def __init__(
        self,
        pretrained: str,
        *,
        image_token_id: Optional[int] = None,
        image_string: Optional[str] = None,
        interleave: bool = True,
        max_images: int = 999,
        convert_img_format: bool = False,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        image_max_side: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.image_width = image_width
        self.image_height = image_height
        self.image_max_side = image_max_side
        if self.image_max_side and (self.image_width or self.image_height):
            raise ValueError(
                "Ambiguous config for image resize: cannot set both "
                "image_max_side and (image_width or image_height)"
            )
        self._processor_pixel_kwargs = {}
        if min_pixels is not None:
            self._processor_pixel_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            self._processor_pixel_kwargs["max_pixels"] = max_pixels

        self.interleave = interleave
        self.max_images = max_images
        self.rgb = convert_img_format

        # RBLNLM.__init__ loads AutoConfig, AutoTokenizer, resolves model_type
        # (which will be "vlm" for VLMs via the new mapping-dict logic), and
        # compiles/loads the RBLN model. The tokenizer is replaced with
        # processor.tokenizer immediately afterward to ensure consistent
        # handling of image placeholder tokens.
        super().__init__(pretrained=pretrained, **kwargs)

        if self.model_type != "vlm":
            raise ValueError(
                f"RBLNVLM expected a vision-language model but got model_type="
                f"'{self.model_type}' for {pretrained}. Use --model rbln for "
                "text-only LLMs."
            )

        self._load_processor(
            pretrained,
            revision=kwargs.get("revision", "main"),
            trust_remote_code=kwargs.get("trust_remote_code", False),
        )

        if image_string:
            self.image_token = image_string
            logger.info(
                f"Using user-provided image placeholder string '{image_string}'."
            )
        else:
            # Some VLM processors (e.g. Gemma3, PaliGemma) wrap the inner
            # image token in a BOI/EOI sequence and only look for the
            # ``boi_token`` in the prompt — the processor itself then
            # expands it to ``boi + image_token*N + eoi``. Prefer that
            # placeholder when available so the processor's image-token
            # accounting matches.
            boi_token = getattr(self.processor.tokenizer, "boi_token", None)
            if boi_token:
                self.image_token = boi_token
                self.image_token_id = getattr(
                    self.processor.tokenizer, "boi_token_id", None
                )
                logger.info(
                    f"Using processor BOI token '{boi_token}' as image placeholder."
                )
            else:
                self.image_token_id = (
                    int(image_token_id)
                    if image_token_id is not None
                    else (
                        getattr(self.config, "image_token_id", None)
                        or getattr(self.config, "image_token_index", None)
                    )
                )
                if self.image_token_id is None:
                    raise ValueError(
                        "Could not infer image_token_id from model config. Pass "
                        "image_token_id explicitly via --model_args."
                    )
                self.image_token = self.tok_decode(
                    [self.image_token_id], skip_special_tokens=False
                )
                logger.info(
                    f"Resolved image_token_id={self.image_token_id} -> '{self.image_token}'"
                )

        self.chat_applied = False

    def _load_processor(
        self, pretrained: str, revision: str, trust_remote_code: bool
    ) -> None:
        self.processor = transformers.AutoProcessor.from_pretrained(
            pretrained,
            revision=revision,
            trust_remote_code=trust_remote_code,
            **self._processor_pixel_kwargs,
        )
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def device(self):
        # RBLN models do not expose .parameters() the same way as torch.nn.Module
        # graphs; pin tensors to CPU and let optimum.rbln handle NPU transfer.
        return torch.device("cpu")

    def tok_decode(self, tokens, skip_special_tokens: bool = True):
        if isinstance(tokens, int):
            tokens = [tokens]
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def tok_batch_multimodal_encode(
        self,
        strings: List[str],
        images: List[List],
        padding_side: str = "left",
        left_truncate_len: Optional[int] = None,
        truncation: bool = False,
    ) -> Union[BatchEncoding, Dict[str, torch.Tensor]]:
        if not self.chat_applied:
            strings = [
                replace_placeholders(
                    s, DEFAULT_IMAGE_PLACEHOLDER, self.image_token, self.max_images
                )
                for s in strings
            ]

        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side

        images = [imgs[: self.max_images] for imgs in images]
        if self.rgb:
            images = [[img.convert("RGB") for img in sub] for sub in images]

        if getattr(self.config, "model_type", "") == "llava":
            images = flatten_image_list(images)

        encoding = self.processor(
            images=images,
            text=strings,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
        )

        if left_truncate_len:
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][
                :, -left_truncate_len:
            ]
        self.tokenizer.padding_side = old_padding_side
        return encoding

    def _model_multimodal_generate(
        self, inputs, stop, **generation_kwargs
    ):
        generation_kwargs.setdefault("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample")
        if generation_kwargs["temperature"] == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False
        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature", None)

        stopping_criteria = stop_sequences_criteria(
            self.tokenizer,
            stop,
            inputs["input_ids"].shape[1],
            inputs["input_ids"].shape[0],
        )
        with torch.inference_mode():
            return self.model.generate(
                **inputs,
                stopping_criteria=stopping_criteria,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                **generation_kwargs,
            )

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        if requests and len(requests[0].args) < 3:
            return super().loglikelihood(requests=requests, disable_tqdm=disable_tqdm)
        raise NotImplementedError(
            "RBLNVLM does not yet implement multimodal loglikelihood. "
            "Use generate_until tasks (e.g. mmmu, chartqa)."
        )

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        if requests and len(requests[0].args) < 3:
            return super().loglikelihood_rolling(
                requests=requests, disable_tqdm=disable_tqdm
            )
        raise NotImplementedError(
            "RBLNVLM does not support multimodal loglikelihood_rolling."
        )

    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        if requests and len(requests[0].args) < 3:
            return super().generate_until(requests=requests, disable_tqdm=disable_tqdm)

        res: List[str] = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests (RBLN VLM)",
        )

        re_ords = Collator(
            [req.args for req in requests],
            _collate,
            group_by="gen_kwargs",
            group_fn=lambda x: x[1],
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        eos = self.tok_decode([self.eot_token_id], skip_special_tokens=False)

        for chunk in chunks:
            contexts, all_gen_kwargs, aux_arguments = zip(*chunk)

            visuals = [
                [
                    resize_image(
                        img,
                        self.image_width,
                        self.image_height,
                        self.image_max_side,
                    )
                    for img in arg["visual"]
                ]
                for arg in aux_arguments
            ]

            contexts = list(contexts)

            gen_kwargs = all_gen_kwargs[0]
            if not isinstance(gen_kwargs, dict):
                raise ValueError(
                    f"Expected `gen_kwargs` dict but got {type(gen_kwargs)}"
                )
            kwargs = copy.deepcopy(gen_kwargs)
            until = handle_stop_sequences(kwargs.pop("until", None), eos=eos)
            max_gen_toks = kwargs.pop("max_gen_toks", self.max_gen_toks)
            max_ctx_len = self.max_length - max_gen_toks

            inputs = self.tok_batch_multimodal_encode(
                contexts,
                visuals,
                left_truncate_len=max_ctx_len,
                truncation=self.truncation,
            )
            context_enc = inputs["input_ids"]

            if "max_length" not in kwargs:
                kwargs["max_length"] = context_enc.shape[1] + max_gen_toks

            cont = self._model_multimodal_generate(inputs, stop=until, **kwargs)

            for cont_toks, context in zip(cont.tolist(), contexts):
                cont_toks = cont_toks[context_enc.shape[1] :]
                s = self.tok_decode(cont_toks)
                for term in until:
                    if term:
                        s = s.split(term)[0]
                res.append(s)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), s)
                pbar.update(1)

        pbar.close()
        return re_ords.get_original(res)
