import importlib
from unittest.mock import MagicMock

import torch

from lm_eval.api.instance import Instance
from lm_eval.api.registry import model_registry


def _import_hf_audiolm(monkeypatch):
    # The lazy mapping for "hf-audiolm-qwen" still names `HFAudioLM` (fixed
    # separately in #4501), so importing the module trips the registry's
    # duplicate-alias check. Point the placeholder at the real class for the
    # duration of this test; once #4501 lands this is a no-op.
    monkeypatch.setitem(
        model_registry._objs,
        "hf-audiolm-qwen",
        "lm_eval.models.hf_audiolm:HFAUDIOLMQWEN",
    )
    return importlib.import_module("lm_eval.models.hf_audiolm").HFAUDIOLMQWEN


def test_generate_until_moves_inputs_to_model_device(monkeypatch):
    """Regression test for #4256: inputs must follow `self.device`, not "cuda"."""
    HFAUDIOLMQWEN = _import_hf_audiolm(monkeypatch)
    moves = []

    class FakeTensor:
        shape = (1, 3)

        def to(self, device):
            moves.append(device)
            return self

    class FakeEncoding(dict):
        pass

    encoding = FakeEncoding(input_ids=FakeTensor())
    encoding.input_ids = encoding["input_ids"]

    model = HFAUDIOLMQWEN.__new__(HFAUDIOLMQWEN)
    model._device = torch.device("cpu")
    model._rank = 0
    model.batch_size_per_gpu = 1
    model._max_length = 64
    model.truncation = False
    model.cache_hook = MagicMock()
    model.tok_encode = lambda s, **kw: [0]
    model.tok_decode = lambda t, **kw: "x"
    model.tok_batch_multimodal_encode = lambda *a, **kw: encoding
    model._model_multimodal_generate = lambda *a, **kw: torch.tensor([[0, 0, 0, 1]])
    monkeypatch.setattr(HFAUDIOLMQWEN, "eot_token_id", 0, raising=False)

    req = Instance(
        "generate_until",
        {},
        ("hi", {"max_gen_toks": 4}, {"audio": [{"array": [0.0, 0.1]}]}),
        0,
    )
    assert model.generate_until([req], disable_tqdm=True) == ["x"]

    assert moves == [model.device]
