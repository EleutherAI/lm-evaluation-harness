"""Tests for VLLM.modify_gen_kwargs. CPU only; no vLLM engine is started."""

import pytest


pytest.importorskip("vllm")

from lm_eval.models.vllm_causallms import VLLM


@pytest.mark.parametrize(
    "gen_kwargs,expected_temperature",
    [
        # neither do_sample nor temperature: greedy, like the HF backend
        ({"until": ["\n"], "max_gen_toks": 48}, 0.0),
        ({"do_sample": False}, 0.0),
        ({"do_sample": False, "temperature": 0.5}, 0.0),
        ({"do_sample": True, "temperature": 0.7}, 0.7),
    ],
)
def test_temperature_reaches_sampling_params(gen_kwargs, expected_temperature):
    kwargs, _, _ = VLLM.modify_gen_kwargs(gen_kwargs, eos="</s>")
    assert "do_sample" not in kwargs
    assert kwargs["temperature"] == expected_temperature
