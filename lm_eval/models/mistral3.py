"""
Mistral3 model adapter for lm-evaluation-harness.

This adapter enables evaluation of Ministral-3 models (3B, 8B, 14B) which use
Mistral3ForConditionalGeneration instead of AutoModelForCausalLM.

Usage:
    lm_eval --model hf-mistral3 \
        --model_args pretrained=mistralai/Ministral-3-3B-Instruct-2512-BF16,dtype=bfloat16 \
        --tasks hellaswag \
        --device cuda:0 \
        --batch_size 8
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import torch

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


if TYPE_CHECKING:
    import transformers

eval_logger = logging.getLogger(__name__)


@register_model("hf-mistral3")
class Mistral3LM(HFLM):
    """
    Model adapter for Mistral3 models (Ministral-3 family).

    These models use Mistral3ForConditionalGeneration which is a vision-language
    model class, but can be used for text-only evaluation by ignoring the vision
    encoder.
    """

    AUTO_MODEL_CLASS = None  # Set dynamically in __init__

    def __init__(self, **kwargs):
        # Import here to avoid import errors if transformers version doesn't support Mistral3
        try:
            from transformers import Mistral3ForConditionalGeneration

            self.AUTO_MODEL_CLASS = Mistral3ForConditionalGeneration
        except ImportError:
            raise ImportError(
                "Mistral3ForConditionalGeneration not found in transformers. "
                "Please install transformers >= 5.0.0 or from main: "
                "pip install git+https://github.com/huggingface/transformers"
            ) from None

        super().__init__(**kwargs)

    def _get_backend(
        self,
        config: transformers.PretrainedConfig | transformers.AutoConfig,
        backend: Literal["default", "causal", "seq2seq"] = "default",
        trust_remote_code: bool | None = False,
    ) -> None:
        """
        Override to force causal backend for Mistral3 models.

        Mistral3 models are decoder-only despite using a conditional generation class.
        """
        # Always use causal backend for Mistral3
        self.backend = "causal"
        eval_logger.info("Using backend 'causal' for Mistral3 model")

    def _model_call(
        self,
        inps: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Override to handle Mistral3 model output format.

        Mistral3ForConditionalGeneration returns logits in the same format as
        causal LMs, so we call the model directly but bypass the base class
        assertion that checks for AutoModelForCausalLM.
        """
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=self.device.type,
                dtype=self.mixed_precision_dtype,
                enabled=self.mixed_precision_dtype is not None,
            ),
        ):
            # Mistral3 models work like causal LMs for text-only input
            return self.model(inps).logits

    @property
    def max_length(self) -> int:
        """Get the maximum sequence length for the model."""
        if self._max_length:
            return self._max_length

        seqlen_config_attrs = (
            "max_position_embeddings",
            "n_positions",
            "n_ctx",
        )

        # First check text_config if it exists (for VLM-style models like Mistral3)
        if hasattr(self.model.config, "text_config"):
            text_config = self.model.config.text_config
            for attr in seqlen_config_attrs:
                if hasattr(text_config, attr):
                    return getattr(text_config, attr)

        # Fall back to main config
        for attr in seqlen_config_attrs:
            if hasattr(self.model.config, attr):
                return getattr(self.model.config, attr)

        # Check tokenizer
        if (
            hasattr(self.tokenizer, "model_max_length")
            and self.tokenizer.model_max_length < 1000000000
        ):
            return self.tokenizer.model_max_length

        return self._DEFAULT_MAX_LENGTH
