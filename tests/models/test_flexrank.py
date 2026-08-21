from unittest.mock import Mock

import pytest

from lm_eval.models.flexrank import FlexRankLM
from lm_eval.models.huggingface import HFLM


@pytest.fixture
def mocked_hflm(monkeypatch):
    model = Mock()
    model.reduce_size = Mock()
    model.virtual_size_ratio = 0.75

    def init(instance, **kwargs):
        instance._model = model

    monkeypatch.setattr(HFLM, "__init__", init)
    return model


def test_selects_size_ratio(mocked_hflm):
    FlexRankLM(pretrained="flexrank-checkpoint", size_ratio=0.75)

    mocked_hflm.reduce_size.assert_called_once_with(
        size_ratio=0.75, compression_rate=None
    )


def test_selects_compression_rate(mocked_hflm):
    FlexRankLM(pretrained="flexrank-checkpoint", compression_rate=0.25)

    mocked_hflm.reduce_size.assert_called_once_with(
        size_ratio=None, compression_rate=0.25
    )


def test_forwards_unspecified_reduction_args(mocked_hflm):
    FlexRankLM(pretrained="flexrank-checkpoint")

    mocked_hflm.reduce_size.assert_called_once_with(
        size_ratio=None, compression_rate=None
    )


def test_select_profile_resets_profile_dependent_state(mocked_hflm):
    lm = FlexRankLM(pretrained="flexrank-checkpoint", size_ratio=0.75)
    mocked_hflm.reduce_size.reset_mock()
    lm.batch_sizes = {0: 8}

    lm.select_profile(size_ratio=0.5)

    mocked_hflm.reduce_size.assert_called_once_with(
        size_ratio=0.5, compression_rate=None
    )
    assert lm.batch_sizes == {}


def test_factory_reuses_active_sweep_model(mocked_hflm, monkeypatch):
    lm = FlexRankLM(pretrained="flexrank-checkpoint", size_ratio=0.75)
    mocked_hflm.reduce_size.reset_mock()
    rng_state = {"state": "captured"}
    restore = Mock()
    monkeypatch.setattr(FlexRankLM, "_sweep_model", lm)
    monkeypatch.setattr(FlexRankLM, "_sweep_rng_state", rng_state)
    monkeypatch.setattr("lm_eval.models.flexrank.restore_rng_state", restore)

    reused = FlexRankLM.create_from_arg_obj({"size_ratio": 0.5})

    assert reused is lm
    mocked_hflm.reduce_size.assert_called_once_with(
        size_ratio=0.5, compression_rate=None
    )
    restore.assert_called_once_with(rng_state)


def test_rejects_non_flexrank_checkpoint(monkeypatch):
    def init(instance, **kwargs):
        instance._model = object()

    monkeypatch.setattr(HFLM, "__init__", init)

    with pytest.raises(TypeError, match="does not provide.*`reduce_size`"):
        FlexRankLM(pretrained="regular-hf-checkpoint")


@pytest.mark.parametrize("parallelize", [False, None])
def test_avoids_device_map_for_single_device_load(monkeypatch, parallelize):
    monkeypatch.setattr(
        HFLM,
        "_get_accelerate_args",
        lambda *args, **kwargs: {"device_map": {"": "cpu"}, "max_memory": None},
    )

    lm = object.__new__(FlexRankLM)

    assert lm._get_accelerate_args(parallelize=parallelize) == {}


def test_preserves_device_map_for_model_parallel_load(monkeypatch):
    expected = {"device_map": "auto", "max_memory": {0: "10GiB"}}
    monkeypatch.setattr(
        HFLM, "_get_accelerate_args", lambda *args, **kwargs: expected.copy()
    )

    lm = object.__new__(FlexRankLM)

    assert lm._get_accelerate_args(parallelize=True) == expected
