from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from lm_eval.models.huggingface import HFLM
from lm_eval.models.mamba_lm import MambaLMWrapper
from lm_eval.models.optimum_habana import HabanaLM


@pytest.mark.parametrize("backend", [HFLM, MambaLMWrapper, HabanaLM])
@pytest.mark.parametrize(
    "kwargs, expected_temperature, expected_sampling",
    [
        ({"do_sample": True}, None, True),
        ({"do_sample": True, "temperature": 0.7}, 0.7, True),
        ({}, None, False),
        ({"do_sample": False}, None, False),
        ({"temperature": 0.0}, None, False),
        ({"do_sample": False, "temperature": 0.0}, None, False),
        ({"temperature": 0.7}, 0.7, None),
    ],
)
def test_generation_temperature_defaults(
    backend, kwargs, expected_temperature, expected_sampling
):
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=16,
            n_positions=8,
            n_layer=1,
            n_embd=8,
            n_head=2,
            bos_token_id=1,
            eos_token_id=None,
            pad_token_id=0,
        )
    ).eval()
    model.generation_config.temperature = 0.65
    generate = model.generate

    def generate_on_cpu(**generation_kwargs):
        # Gaudi's extra flags do not apply to the CPU model used in this test.
        generation_kwargs.pop("hpu_graphs", None)
        generation_kwargs.pop("lazy_mode", None)
        return generate(**generation_kwargs)

    model.generate = Mock(side_effect=generate_on_cpu)
    wrapper = SimpleNamespace(
        model=model,
        tokenizer=SimpleNamespace(pad_token_id=0),
        device=torch.device("cpu"),
        mixed_precision_dtype=None,
        is_hf=True,
        lazy_mode=False,
        find_bucket=lambda length: length,
    )
    context = torch.tensor([[1, 3, 4]])
    if backend is HabanaLM:
        kwargs = {**kwargs, "attention_mask": torch.ones_like(context)}

    output = backend._model_generate(wrapper, context, max_length=5, stop=[], **kwargs)

    assert output.shape == (1, 5)
    sent = model.generate.call_args.kwargs
    if expected_temperature is None:
        assert "temperature" not in sent
    else:
        assert sent["temperature"] == expected_temperature
    assert sent.get("do_sample") is expected_sampling
    assert model.generation_config.temperature == 0.65


def test_explicit_zero_temperature_sampling_remains_invalid():
    from transformers import TemperatureLogitsWarper

    def generate(**kwargs):
        TemperatureLogitsWarper(kwargs["temperature"])

    wrapper = SimpleNamespace(
        model=SimpleNamespace(generate=generate),
        tokenizer=SimpleNamespace(pad_token_id=0),
        device=torch.device("cpu"),
        mixed_precision_dtype=None,
    )
    with pytest.raises(ValueError, match="strictly positive"):
        HFLM._model_generate(
            wrapper,
            torch.tensor([[1, 3]]),
            max_length=4,
            stop=[],
            do_sample=True,
            temperature=0.0,
        )
