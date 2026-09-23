"""Inference-only MXFP4 reference arithmetic implemented with PyTorch tensors.

The format follows OCP MX v1.0, sections 5.3.3, 5.4.1 and 6.3. This module
implements the floor shared-exponent recipe, not a packed/native FP4 kernel.
"""

from __future__ import annotations

import fnmatch
from collections import defaultdict
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F


if TYPE_CHECKING:
    from collections.abc import Sequence

BLOCK_SIZE = 32
FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@torch.no_grad()
def mxfp4_qdq(tensor: torch.Tensor) -> torch.Tensor:
    """Round last-dimension blocks to E2M1/E8M0, returning the input dtype.

    Each block uses 2**(floor(log2(amax)) - 2), clamped to the E8M0
    exponent range [-127, 127]. Elements use round-to-nearest, ties-to-even
    and saturate at +/-6. Nonfinite blocks become NaN; zero is canonical +0.
    Exponents are extracted with frexp to avoid log2 rounding at powers of two.
    """
    if tensor.dtype not in FLOAT_DTYPES:
        raise TypeError("MXFP4 reference QDQ requires float16, bfloat16 or float32")
    if tensor.ndim == 0 or tensor.shape[-1] == 0 or tensor.shape[-1] % BLOCK_SIZE:
        raise ValueError("MXFP4 requires a nonempty last dimension divisible by 32")

    blocks = tensor.float().reshape(*tensor.shape[:-1], -1, BLOCK_SIZE)
    maximum = blocks.abs().amax(dim=-1, keepdim=True)
    finite = torch.isfinite(maximum)
    safe_maximum = torch.where(finite & (maximum > 0), maximum, 1.0)
    _, exponent = torch.frexp(safe_maximum)
    exponent = (exponent - 3).clamp(-127, 127)
    exponent = torch.where(maximum == 0, -127, exponent)
    scale = torch.ldexp(torch.ones_like(maximum), exponent)

    normalized = torch.where(finite, blocks, 0.0) / scale
    magnitude = normalized.abs().clamp(max=6.0)
    _, element_exponent = torch.frexp(magnitude)
    step = torch.ldexp(torch.ones_like(magnitude), (element_exponent - 2).clamp(min=-1))
    rounded = torch.round(magnitude / step) * step
    restored = rounded * normalized.sign() * scale
    restored = torch.where(restored == 0, 0.0, restored)
    restored = torch.where(finite, restored, float("nan"))
    return restored.reshape(tensor.shape).to(tensor.dtype)


class MXFP4Linear(nn.Linear):
    """An inference Linear with static QDQ weights and optional dynamic A4.

    Reuses the original parameters without a second model-sized allocation.
    Conversion overwrites the selected weights and is therefore irreversible;
    reload the original checkpoint for a floating-point baseline.
    """

    def __init__(self, source: nn.Linear, *, activation_quantized: bool):
        super().__init__(
            source.in_features,
            source.out_features,
            bias=source.bias is not None,
            device="meta",
            dtype=source.weight.dtype,
        )
        self.weight = source.weight
        self.bias = source.bias
        self.activation_quantized = activation_quantized
        self.training = False
        with torch.no_grad():
            self.weight.copy_(mxfp4_qdq(self.weight))

    def train(self, mode: bool = True):
        if mode:
            raise RuntimeError(
                "MXFP4Linear is inference-only; reload the original checkpoint to train"
            )
        return super().train(False)

    @torch.no_grad()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.activation_quantized:
            input = mxfp4_qdq(input)
        return F.linear(input, self.weight, self.bias)


def convert_mxfp4(
    model: nn.Module, *, precision: str, ignore: Sequence[str]
) -> dict[str, list[str]]:
    """Validate then replace plain Linear layers, reporting exact coverage.

    Shared module aliases are preserved. Shared parameter storage across
    distinct modules is rejected to avoid changing an excluded embedding/head.
    Hooks and specialized Linear subclasses require a dedicated integration.
    """
    if precision not in ("w4a4", "w4a16"):
        raise ValueError("MXFP4 conversion requires w4a4 or w4a16")
    modules = list(model.named_modules(remove_duplicate=False))
    if any(isinstance(module, MXFP4Linear) for _, module in modules):
        raise ValueError(
            "Model already contains MXFP4 layers; reload the original checkpoint"
        )
    if any(hasattr(module, "_hf_hook") for _, module in modules):
        raise ValueError("hf-mxfp4 does not support Accelerate dispatch/offload hooks")

    output = (
        model.get_output_embeddings()
        if hasattr(model, "get_output_embeddings")
        else None
    )
    aliases = defaultdict(list)
    for name, module in modules:
        if isinstance(module, nn.Linear):
            aliases[id(module)].append((name, module))

    # Include buffers and all direct parameters, including those on ignored modules.
    owners = defaultdict(set)
    for _, module in modules:
        for value in list(module.parameters(recurse=False)) + list(
            module.buffers(recurse=False)
        ):
            if value.device.type == "meta":
                raise ValueError(
                    "hf-mxfp4 requires materialized weights; meta/offload is unsupported"
                )
            if value.numel():
                owners[(str(value.device), value.untyped_storage().data_ptr())].add(
                    id(module)
                )

    candidates = []
    report: dict[str, list[str]] = {"quantized_modules": [], "excluded_modules": []}
    for entries in aliases.values():
        names = [name for name, _ in entries]
        module = entries[0][1]
        if module is output or any(
            fnmatch.fnmatchcase(name, pattern) for name in names for pattern in ignore
        ):
            report["excluded_modules"].extend(names)
            continue
        if type(module) is not nn.Linear:
            raise ValueError(
                f"Unsupported Linear subclass at {names[0]}; explicitly exclude it if intended"
            )
        if (
            not names[0]
            or module.in_features == 0
            or module.out_features == 0
            or module.in_features % BLOCK_SIZE
        ):
            raise ValueError(
                f"Linear {names[0]!r} must be a child module with in_features divisible by 32"
            )
        if module.weight.dtype not in FLOAT_DTYPES:
            raise ValueError(
                f"Linear {names[0]} is not a supported floating-point weight"
            )
        if module._forward_hooks or module._forward_pre_hooks:
            raise ValueError(
                f"Linear {names[0]} has hooks that cannot be preserved by conversion"
            )
        key = (str(module.weight.device), module.weight.untyped_storage().data_ptr())
        if owners[key] != {id(module)}:
            raise ValueError(
                f"Linear {names[0]} shares weight storage with another module; explicitly exclude it"
            )
        if not torch.isfinite(module.weight).all():
            raise ValueError(f"Linear {names[0]} contains nonfinite weights")
        candidates.append(entries)

    if not candidates:
        raise ValueError(
            "No eligible Linear layers: hf-mxfp4 cannot evaluate an unquantized model as W4"
        )

    for entries in candidates:
        replacement = MXFP4Linear(
            entries[0][1], activation_quantized=precision == "w4a4"
        )
        for name, _ in entries:
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, replacement)
            report["quantized_modules"].append(name)
    return report
