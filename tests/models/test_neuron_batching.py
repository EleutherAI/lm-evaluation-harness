"""Exercise Neuron's generation wrapper on CPU, before and after device execution."""

from importlib.util import find_spec, module_from_spec
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from transformers import GenerationConfig


@pytest.fixture(scope="module")
def neuron_adapter():
    # Isolate the adapter's import-time registration from the shared model registry.
    spec = find_spec("lm_eval.models.neuron_optimum")
    module = module_from_spec(spec)
    with patch("lm_eval.api.registry.register_model", return_value=lambda cls: cls):
        spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("with_mask", [True, False], ids=["mask", "no-mask"])
def test_short_static_batch(neuron_adapter, monkeypatch, with_mask):
    input_ids = torch.tensor([[11, 12, 13], [99, 21, 22]])
    attention_mask = torch.tensor([[1, 1, 1], [0, 1, 1]]) if with_mask else None
    # Give all four device rows distinct outputs so a leaked padding row is visible.
    device_output = torch.arange(16).reshape(4, 4)
    model = Mock(
        batch_size=4,
        max_length=8,
        continuous_batching=False,
        config=SimpleNamespace(eos_token_id=99),
        generate_tokens=Mock(return_value=device_output),
    )
    selector = Mock()
    # The module has no TokenSelector binding when optimum-neuron is absent.
    monkeypatch.setattr(neuron_adapter, "TokenSelector", selector, raising=False)

    output = neuron_adapter.CustomNeuronModelForCausalLM.generate(
        model,
        input_ids,
        attention_mask=attention_mask,
        generation_config=GenerationConfig(),
    )

    model.generate_tokens.assert_called_once()
    args, kwargs = model.generate_tokens.call_args
    padded_ids, _, real_batch_size = args
    assert real_batch_size == 2
    assert torch.equal(
        padded_ids,
        torch.tensor([[11, 12, 13], [99, 21, 22], [99, 99, 99], [99, 99, 99]]),
    )
    if with_mask:
        assert torch.equal(
            kwargs["attention_mask"],
            torch.tensor([[1, 1, 1], [0, 1, 1], [0, 0, 0], [0, 0, 0]]),
        )
    else:
        assert kwargs["attention_mask"] is None
    assert torch.equal(output, device_output[:2])
