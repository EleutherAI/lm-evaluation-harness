import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.models.megatron_lm import MegatronLMEval


@pytest.fixture
def model():
    model = MegatronLMEval.__new__(MegatronLMEval)
    LM.__init__(model)
    model._max_length = 12
    model._max_gen_toks = 8
    model._batch_size = 2
    model._device = torch.device("cpu")
    model._global_rank = 0
    model._rank = 0
    model._world_size = 1
    model._ep_size = 1
    model._parallelism_mode = "single"
    model.tokenizer = SimpleNamespace(
        eod=0,
        tokenize=lambda text: [ord(c) for c in text],
        detokenize=lambda ids: "".join(chr(i) for i in ids if i),
    )
    model.prompt_calls = []
    model.cache_hook = Mock()

    def forward(input_ids, attention_mask=None, position_ids=None):
        model.prompt_calls.append(
            ["".join(chr(i) for i in row if i) for row in input_ids.tolist()]
        )
        logits = torch.zeros(*input_ids.shape, 128)
        logits[:, :, ord("A")] = 10.0
        return logits

    model._model_forward = forward
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
