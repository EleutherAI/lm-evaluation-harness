import pytest

from lm_eval.models.huggingface import HFLM
from lm_eval.models.optimum_habana import HabanaLM


def test_generate_until_restores_max_length_after_failure(monkeypatch):
    monkeypatch.setattr(HFLM, "max_length", property(lambda self: 2048))

    def fail_generation(self, requests, disable_tqdm=False):
        assert self.max_length == 2048
        raise RuntimeError("simulated generation failure")

    monkeypatch.setattr(HFLM, "generate_until", fail_generation)

    model = object.__new__(HabanaLM)
    model._max_length = None
    model.buckets = [128]

    with pytest.raises(RuntimeError, match="simulated generation failure"):
        model.generate_until([])

    assert model.max_length == 128
