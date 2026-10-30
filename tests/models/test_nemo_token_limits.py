import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.models.nemo_lm import NeMoLM


@pytest.fixture
def model():
    model = NeMoLM.__new__(NeMoLM)
    LM.__init__(model)
    model._max_length = 12
    model._max_gen_toks = 8
    model._batch_size = 2
    model.model = object()
    model.tokenizer = SimpleNamespace(
        eos_id=0,
        text_to_ids=lambda text: [ord(c) for c in text],
        ids_to_text=lambda ids: "".join(chr(i) for i in ids),
        ids_to_tokens=lambda ids: ["<eos>"],
    )
    model.prompt_calls = []
    model.cache_hook = Mock()

    def generate(model_arg, inputs, tokens_to_generate, end_strings, greedy):
        model.prompt_calls.append(inputs)
        assert greedy is True
        assert "<eos>" in end_strings
        return {"sentences": [ctx + "A" * tokens_to_generate for ctx in inputs]}

    model.generate = generate
    return model


@pytest.mark.parametrize(
    "options,expected",
    [
        ({}, 8),
        ({"max_gen_toks": 3}, 3),
        ({"max_new_tokens": 3}, 3),
        ({"max_tokens": 3}, 3),
        ({"max_completion_tokens": 3}, 3),
        ({"max_new_tokens": "3"}, 3),
        ({"max_gen_toks": 2, "max_new_tokens": 3, "max_tokens": 4}, 2),
        ({"max_new_tokens": 3, "max_tokens": 4, "max_completion_tokens": 5}, 3),
        ({"max_tokens": 4, "max_completion_tokens": 5}, 4),
        ({"max_gen_toks": None, "max_new_tokens": 3}, 3),
    ],
)
def test_generation_token_aliases(model, options, expected):
    original = copy.deepcopy(options)
    request = Instance("generate_until", {}, ("0123456789", options), 0)

    assert model.generate_until([request]) == ["A" * expected]
    assert model.prompt_calls[0] == ["0123456789"[-(12 - expected) :]]
    assert options == original


def test_aliases_preserve_order_and_kwargs_across_batches(model):
    contexts = ["x", "long prompt", "mid"]
    options = [{"max_new_tokens": 2}, {"max_tokens": 3}, {"max_new_tokens": 2}]
    original = copy.deepcopy(options)
    requests = [
        Instance("generate_until", {}, (ctx, opts), i)
        for i, (ctx, opts) in enumerate(zip(contexts, options, strict=True))
    ]

    assert model.generate_until(requests) == ["AA", "AAA", "AA"]
    assert options == original
    cached = {
        call.args[1][0]: call.args[2]
        for call in model.cache_hook.add_partial.call_args_list
    }
    assert cached == dict(zip(contexts, ["AA", "AAA", "AA"], strict=True))
