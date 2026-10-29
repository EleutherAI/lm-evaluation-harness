"""The ONNX backends preload pip's CUDA libraries, only when a CUDA provider is used."""

import sys
import types

import pytest

from lm_eval.models.onnxruntime_genai import ONNXRuntimeGenAILM
from lm_eval.models.onnxruntime_ort import ONNXRuntimeLM


BACKENDS = [ONNXRuntimeLM, ONNXRuntimeGenAILM]


@pytest.fixture
def ort(monkeypatch):
    """A stand-in ``onnxruntime`` that counts ``preload_dlls()`` calls."""
    fake = types.SimpleNamespace(calls=0)

    def preload_dlls():
        fake.calls += 1

    fake.preload_dlls = preload_dlls
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    return fake


def _model(cls, execution_provider, declared=()):
    """A backend instance with just what provider resolution reads (no model load)."""
    model = cls.__new__(cls)
    model.execution_provider = execution_provider
    options = [{name: {}} for name in declared]
    model._genai_config = {
        "model": {"decoder": {"session_options": {"provider_options": options}}}
    }
    return model


@pytest.mark.parametrize("cls", BACKENDS)
@pytest.mark.parametrize(
    "provider",
    ["cuda", "CUDA", "CUDAExecutionProvider", "tensorrt", "trt", "NvTensorRtRtx"],
)
def test_a_cuda_provider_preloads_the_cuda_libraries(ort, cls, provider):
    _model(cls, provider)._preload_cuda_libraries()
    assert ort.calls == 1


@pytest.mark.parametrize("cls", BACKENDS)
@pytest.mark.parametrize(
    "provider", ["cpu", "CPUExecutionProvider", "dml", "migraphx", "rocm", "webgpu"]
)
def test_other_providers_do_not_preload(ort, cls, provider):
    _model(cls, provider)._preload_cuda_libraries()
    assert ort.calls == 0


@pytest.mark.parametrize("cls", BACKENDS)
@pytest.mark.parametrize(
    ("declared", "calls"),
    [(("cuda",), 1), (("NvTensorRtRtx",), 1), (("dml",), 0), ((), 0)],
)
def test_without_execution_provider_the_exports_declared_providers_decide(
    ort, cls, declared, calls
):
    _model(cls, None, declared)._preload_cuda_libraries()
    assert ort.calls == calls
