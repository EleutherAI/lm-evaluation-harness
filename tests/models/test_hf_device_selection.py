"""Test HFLM device selection without loading models or accelerator runtimes."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lm_eval.models import huggingface


class DeviceSelected(Exception):
    """Stop initialization before config, tokenizer, or model loading."""


@pytest.mark.parametrize("accelerator_type", ["xpu", "hpu", "npu"])
@pytest.mark.parametrize("indexed", [False, True])
def test_hf_preserves_requested_accelerator(monkeypatch, accelerator_type, indexed):
    """Both bare and indexed device names select the requested accelerator."""
    accelerator = SimpleNamespace(
        num_processes=1, device=SimpleNamespace(type=accelerator_type)
    )
    monkeypatch.setattr(huggingface, "Accelerator", lambda **kwargs: accelerator)
    monkeypatch.setattr(
        torch, accelerator_type, SimpleNamespace(device_count=lambda: 1), raising=False
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # CPU torch lacks the vendor NPU device registration. Model placement is
    # outside this test; record which device the actual constructor selects.
    device_factory = Mock(side_effect=lambda name: SimpleNamespace(name=name))
    monkeypatch.setattr(torch, "device", device_factory)
    monkeypatch.setattr(
        huggingface.HFLM, "_get_config", Mock(side_effect=DeviceSelected)
    )
    device = f"{accelerator_type}:0" if indexed else accelerator_type
    model = huggingface.HFLM.__new__(huggingface.HFLM)
    with pytest.raises(DeviceSelected):
        model.__init__(pretrained="unused", device=device)
    device_factory.assert_called_once_with(device)
    assert model.device.name == device


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:0", "mps", "mps:0", ""])
def test_hf_standard_device_selection(monkeypatch, device):
    """Existing device names and the CPU fallback retain their behavior."""
    accelerator = SimpleNamespace(num_processes=1, device=SimpleNamespace(type="cuda"))
    monkeypatch.setattr(huggingface, "Accelerator", lambda **kwargs: accelerator)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        huggingface.HFLM, "_get_config", Mock(side_effect=DeviceSelected)
    )
    model = huggingface.HFLM.__new__(huggingface.HFLM)
    with pytest.raises(DeviceSelected):
        model.__init__(pretrained="unused", device=device)
    assert str(model.device) == (device or "cpu")
