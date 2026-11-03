from unittest.mock import Mock

import pytest

from lm_eval.models import megatron_lm


@pytest.mark.parametrize(
    "model_args,additional,expected",
    [
        ("", {}, 1),
        (",micro_batch_size=3", {}, 3),
        ("", {"batch_size": 8}, 8),
        ("", {"batch_size": "8"}, 8),
        (",batch_size=4", {}, 4),
        (",micro_batch_size=3", {"batch_size": 8}, 8),
        (",micro_batch_size=3", {"batch_size": None}, 3),
    ],
)
def test_batch_size_reaches_adapter_and_runtime(
    tmp_path, monkeypatch, model_args, additional, expected
):
    monkeypatch.setattr(megatron_lm, "_add_megatron_to_path", Mock())
    initialize = Mock()
    monkeypatch.setattr(megatron_lm.MegatronLMEval, "_initialize_megatron", initialize)

    model = megatron_lm.MegatronLMEval.create_from_arg_string(
        f"load={tmp_path}{model_args}", additional
    )

    assert model.batch_size == expected
    assert initialize.call_args.kwargs["micro_batch_size"] == expected


@pytest.mark.parametrize("batch_size", ["auto", "auto:4", 0, -1])
def test_invalid_batch_size_fails_before_runtime_initialization(
    tmp_path, monkeypatch, batch_size
):
    initialize = Mock()
    monkeypatch.setattr(megatron_lm, "_add_megatron_to_path", Mock())
    monkeypatch.setattr(megatron_lm.MegatronLMEval, "_initialize_megatron", initialize)

    with pytest.raises(ValueError, match="positive integer"):
        megatron_lm.MegatronLMEval(load=str(tmp_path), batch_size=batch_size)

    initialize.assert_not_called()
