"""CPU coverage of Neuron's generation stopping boundary."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch


@pytest.fixture
def neuron_backend():
    """Load the backend without altering the global model registry."""
    # The existing neuronx lazy registry entry names a different class.
    spec = importlib.util.spec_from_file_location(
        "neuron_stopping_backend",
        Path(__file__).parents[2] / "lm_eval/models/neuron_optimum.py",
    )
    module = importlib.util.module_from_spec(spec)
    with patch("lm_eval.api.registry.register_model", return_value=lambda cls: cls):
        spec.loader.exec_module(module)
    return module.NEURON_HF


class CharacterTokenizer:
    """Keep prompt and generated characters distinguishable without downloads."""

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, tokens):
        return "".join(chr(int(token)) for token in tokens)

    def batch_decode(self, batch):
        return [self.decode(tokens) for tokens in batch]


@pytest.mark.parametrize("prompt", ["Q\n", "Question: what?\nAnswer:\n"])
@pytest.mark.parametrize("stop", ["\n", "END"])
def test_neuron_stop_matches_only_generated_text(neuron_backend, prompt, stop):
    """Prompt stop strings must not halt generation, but generated ones must."""
    tokenizer = CharacterTokenizer()
    prompt += stop
    context = torch.tensor([tokenizer.encode(prompt)])
    captured = {}

    def generate(**kwargs):
        captured.update(kwargs)
        return kwargs["input_ids"]

    model = SimpleNamespace(
        tokenizer=tokenizer,
        config=SimpleNamespace(eos_token_id=0),
        eot_token_id=0,
        model=SimpleNamespace(generate=generate),
    )
    result = neuron_backend._model_generate(
        model, context, max_length=context.shape[1] + 8, stop=[stop]
    )
    assert result is context
    criteria = captured["stopping_criteria"]
    generated = torch.tensor([tokenizer.encode("x")])
    assert not criteria(torch.cat([context, generated], dim=1), None).any()
    generated = torch.tensor([tokenizer.encode("x" + stop)])
    assert criteria(torch.cat([context, generated], dim=1), None).all()


def test_neuron_stop_tracks_each_batch_row(neuron_backend):
    """One row's generated stop must not stop an unfinished row."""
    tokenizer = CharacterTokenizer()
    context = torch.tensor([tokenizer.encode("Q\n"), tokenizer.encode("R\n")])
    captured = {}

    def generate(**kwargs):
        captured.update(kwargs)
        return kwargs["input_ids"]

    model = SimpleNamespace(
        tokenizer=tokenizer,
        config=SimpleNamespace(eos_token_id=0),
        eot_token_id=0,
        model=SimpleNamespace(generate=generate),
    )
    neuron_backend._model_generate(model, context, max_length=8, stop=["\n"])
    criteria = captured["stopping_criteria"]
    generated = torch.tensor([tokenizer.encode("\nx"), tokenizer.encode("xx")])
    assert not criteria(torch.cat([context, generated], dim=1), None).any()
    generated = torch.tensor([tokenizer.encode("\nxx"), tokenizer.encode("xx\n")])
    assert criteria(torch.cat([context, generated], dim=1), None).all()
