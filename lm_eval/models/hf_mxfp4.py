"""Hugging Face MXFP4 fake-quantization backend for accuracy evaluation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils_mxfp4 import BLOCK_SIZE, convert_mxfp4


if TYPE_CHECKING:
    from collections.abc import Sequence

    from transformers import PreTrainedModel

eval_logger = logging.getLogger(__name__)
SUPPORTED_MODEL_TYPES = ("llama", "qwen2", "qwen3", "gpt_neox")


@register_model("hf-mxfp4")
class HFMXFP4(HFLM):
    """Evaluate floating checkpoints through inference-only MXFP4 Linear layers.

    Uses existing HFLM scoring, generation, batching and data parallelism.
    See docs/mxfp4.md for the quantization contract and supported configurations.
    """

    def __init__(
        self,
        pretrained: str | PreTrainedModel,
        precision: str = "w4a4",
        ignore: str | Sequence[str] = "lm_head;*.gate;*.router",
        dtype: str | torch.dtype = "bfloat16",
        device: str = "cuda",
        **kwargs,
    ):
        if precision not in ("bf16", "w4a4", "w4a16"):
            raise ValueError(
                "precision must be bf16, w4a4 or w4a16 (MXFP4, not NVFP4/INT4)"
            )
        if dtype not in (
            "bfloat16",
            "float16",
            "float32",
            torch.bfloat16,
            torch.float16,
            torch.float32,
        ):
            raise ValueError(
                "hf-mxfp4 requires an explicit bfloat16, float16 or float32 dtype"
            )
        if precision == "bf16" and dtype not in ("bfloat16", torch.bfloat16):
            raise ValueError("precision=bf16 requires dtype=bfloat16")
        unsupported = (
            "parallelize",
            "tp_plan",
            "device_map",
            "max_cpu_memory",
            "offload_folder",
            "peft",
            "delta",
            "autogptq",
            "gptqmodel",
            "gguf_file",
            "quantization_config",
            "load_in_4bit",
            "load_in_8bit",
            "mixed_precision_dtype",
        )
        for option in unsupported:
            if kwargs.get(option) is not None and kwargs[option] is not False:
                raise ValueError(
                    f"hf-mxfp4 does not support {option}; use a full floating checkpoint per device"
                )
        if isinstance(ignore, str):
            ignore = [
                pattern.strip() for pattern in ignore.split(";") if pattern.strip()
            ]
        if not isinstance(ignore, (list, tuple)) or not all(
            isinstance(pattern, str) for pattern in ignore
        ):
            raise ValueError(
                "ignore must be a semicolon-separated string or a list of glob patterns"
            )
        self.precision = precision
        self.ignore = list(ignore)
        if device.startswith("npu"):
            try:
                import torch_npu  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "NPU evaluation requires a matching torch_npu installation"
                ) from exc
        if not isinstance(pretrained, str):
            self._validate_config(pretrained.config)
            expected_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
            if any(
                parameter.dtype != expected_dtype
                for parameter in pretrained.parameters()
            ):
                raise ValueError(
                    "A preinitialized model must already use the requested dtype"
                )
        super().__init__(pretrained=pretrained, device=device, dtype=dtype, **kwargs)
        if precision == "bf16" and any(
            parameter.dtype != torch.bfloat16 for parameter in self.model.parameters()
        ):
            raise ValueError(
                "precision=bf16 requires all model parameters to be bfloat16"
            )
        if precision != "bf16":
            self.quantization_report = convert_mxfp4(
                self.model, precision=precision, ignore=self.ignore
            )
        else:
            self.quantization_report = {"quantized_modules": [], "excluded_modules": []}
        eval_logger.info(
            "MXFP4 reference evaluation: precision=%s, quantized=%d, excluded=%d; floating-point storage and matmul",
            precision,
            len(self.quantization_report["quantized_modules"]),
            len(self.quantization_report["excluded_modules"]),
        )

    @staticmethod
    def _validate_config(config):
        if getattr(config, "pretraining_tp", 1) != 1:
            raise ValueError(
                "hf-mxfp4 requires pretraining_tp=1 so Linear inputs are quantized"
            )
        if getattr(config, "quantization_config", None) is not None:
            raise ValueError(
                "hf-mxfp4 requires an unquantized floating-point checkpoint"
            )
        if config.model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"hf-mxfp4 supports dense {SUPPORTED_MODEL_TYPES}, got {config.model_type!r}"
            )

    def _get_config(self, *args, **kwargs):
        super()._get_config(*args, **kwargs)
        self._validate_config(self.config)

    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info["mxfp4"] = {
            "precision": self.precision,
            "execution": "floating_baseline"
            if self.precision == "bf16"
            else "fake_quant_floating_matmul",
            "weight_format": "bfloat16" if self.precision == "bf16" else "mxfp4_e2m1",
            "activation_format": "mxfp4_e2m1"
            if self.precision == "w4a4"
            else str(self.model.dtype),
            "block_size": BLOCK_SIZE if self.precision != "bf16" else None,
            "scale_format": "e8m0" if self.precision != "bf16" else None,
            "scale_policy": "floor" if self.precision != "bf16" else None,
            "rounding": "nearest_ties_to_even" if self.precision != "bf16" else None,
            "ignore": self.ignore,
            **self.quantization_report,
        }
        return info
