from types import SimpleNamespace

import pytest

from lm_eval.models.megatron_lm import MegatronLMEval


@pytest.mark.parametrize("dimension", ["devices", "tp", "pp", "ep"])
@pytest.mark.parametrize("value", [0, -1])
def test_nonpositive_parallelism_is_rejected(dimension, value):
    config = {"devices": 1, "tp": 1, "pp": 1, "ep": 1}
    config[dimension] = value

    with pytest.raises(ValueError):
        MegatronLMEval._validate_parallelism_config(SimpleNamespace(), **config)


@pytest.mark.parametrize(
    "devices, tp, ep, expected_mode",
    [
        (1, 1, 1, "single"),
        (4, 1, 1, "data_parallel"),
        (4, 4, 1, "tensor_parallel"),
        (4, 1, 4, "data_parallel"),
    ],
)
def test_supported_parallelism_modes(devices, tp, ep, expected_mode):
    model = SimpleNamespace()

    MegatronLMEval._validate_parallelism_config(model, devices, tp, 1, ep)

    assert model._parallelism_mode == expected_mode


@pytest.mark.parametrize(
    "devices, tp, pp, ep, error, message",
    [
        (4, 1, 2, 1, AssertionError, "Pipeline Parallelism"),
        (4, 2, 1, 1, ValueError, "TP must equal devices"),
        (4, 1, 1, 2, ValueError, "devices must equal"),
        (4, 2, 1, 4, ValueError, "cannot be combined"),
    ],
)
def test_unsupported_positive_parallelism_is_still_rejected(
    devices, tp, pp, ep, error, message
):
    with pytest.raises(error, match=message):
        MegatronLMEval._validate_parallelism_config(
            SimpleNamespace(), devices, tp, pp, ep
        )
