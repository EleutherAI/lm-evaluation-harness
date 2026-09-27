"""Tensor-only clamp tests, independent of the skipped sparse-model tests."""

import pytest
import torch

from lm_eval.models.hf_steered import SteeredModel


@pytest.mark.parametrize(
    "shape,head_index", [((2, 5, 8, 4), 2), ((2, 5, 1, 4), 0), ((1, 1, 1, 4), 0)]
)
@pytest.mark.parametrize("with_bias", [False, True])
def test_clamp_selected_head(shape, head_index, with_bias):
    acts = (
        torch.arange(torch.Size(shape).numel(), dtype=torch.float64).reshape(shape) / 8
    )
    original = acts.clone()
    direction = torch.tensor([0.6, 0.8, 0.0, 0.0], dtype=acts.dtype)
    orthogonal = torch.tensor([0.8, -0.6, 0.0, 0.0], dtype=acts.dtype)
    bias = torch.arange(4, dtype=acts.dtype) / 8 if with_bias else None
    target = 1.25

    result = SteeredModel.clamp(acts, direction, target, head_index, bias)

    selected = result[:, :, head_index, :]
    centered = selected - bias if bias is not None else selected
    torch.testing.assert_close(
        centered @ direction, torch.full(shape[:2], target, dtype=acts.dtype)
    )
    torch.testing.assert_close(
        selected @ orthogonal, original[:, :, head_index, :] @ orthogonal
    )
    torch.testing.assert_close(
        selected[..., 2:], original[:, :, head_index, 2:], rtol=0, atol=0
    )
    other_heads = [index for index in range(shape[2]) if index != head_index]
    torch.testing.assert_close(
        result[:, :, other_heads, :], original[:, :, other_heads, :], rtol=0, atol=0
    )
    torch.testing.assert_close(acts, original, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(2, 5, 4), (2, 5, 8, 4)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_clamp_whole_layer(shape, with_bias):
    acts = (
        torch.arange(torch.Size(shape).numel(), dtype=torch.float64).reshape(shape) / 8
    )
    original = acts.clone()
    direction = torch.tensor([0.6, 0.8, 0.0, 0.0], dtype=acts.dtype)
    orthogonal = torch.tensor([0.8, -0.6, 0.0, 0.0], dtype=acts.dtype)
    bias = torch.arange(4, dtype=acts.dtype) / 8 if with_bias else None
    target = -1.25

    result = SteeredModel.clamp(acts, direction, target, head_index=None, bias=bias)

    centered = result - bias if bias is not None else result
    torch.testing.assert_close(
        centered @ direction, torch.full(shape[:-1], target, dtype=acts.dtype)
    )
    torch.testing.assert_close(result @ orthogonal, original @ orthogonal)
    torch.testing.assert_close(result[..., 2:], original[..., 2:], rtol=0, atol=0)
    torch.testing.assert_close(acts, original, rtol=0, atol=0)
