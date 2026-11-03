from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.models.megatron_lm import MegatronLMEval


@pytest.mark.parametrize("devices,tp", [(2, 2), (1, 1), (2, 1)])
@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_generation_uses_broadcast_tokens_only_for_tensor_parallelism(
    monkeypatch, devices, tp, temperature
):
    model = MegatronLMEval.__new__(MegatronLMEval)
    LM.__init__(model)
    model._validate_parallelism_config(devices=devices, tp=tp, pp=1, ep=1)
    model._max_length = 12
    model._max_gen_toks = 3
    model._batch_size = 1
    model._device = torch.device("cpu")
    model._global_rank = 0
    model._rank = 0
    model._world_size = 1
    model._ep_size = 1
    model.tokenizer = SimpleNamespace(
        eod=0,
        tokenize=lambda text: [ord(c) for c in text],
        detokenize=lambda ids: "".join(chr(i) for i in ids if i),
    )
    inputs = []

    def forward(input_ids, **kwargs):
        inputs.append(input_ids.clone())
        logits = torch.full((*input_ids.shape, 128), -float("inf"))
        logits[:, :, ord("A")] = 0
        return logits

    def broadcast(tokens, src):
        assert src == 0
        tokens.fill_(ord("B"))

    model._model_forward = forward
    broadcast_mock = Mock(side_effect=broadcast)
    monkeypatch.setattr(torch.distributed, "broadcast", broadcast_mock)
    request = Instance(
        "generate_until", {}, ("hi", {"max_gen_toks": 3, "temperature": temperature}), 0
    )

    expected = "B" if tp > 1 else "A"
    assert model.generate_until([request], disable_tqdm=True) == [expected * 3]
    assert broadcast_mock.call_count == (3 if tp > 1 else 0)
    assert inputs[1][0, -1].item() == ord(expected)
