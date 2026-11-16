"""Cross-backend parity tests: onnxruntime-genai vs raw onnxruntime.

Both ONNX backends read the same Model Builder export and, on the same
execution provider, ultimately call the same ORT kernels -- ``onnxruntime-genai``
only manages the KV cache / position bookkeeping around them. So identical
inputs must yield identical log-probs to floating-point tolerance, and each
backend acts as the other's oracle: a divergence is either a bug in the manual
input construction or a genuine EP/precision difference.

The parity tests need both runtimes plus the Model Builder, and they build a tiny
random Llama export on the fly (CPU execution provider), mirroring
``tests/models/test_onnxruntime_genai.py``; they skip when those are missing, as
they are in CI. The registration and provider-resolution tests need neither
runtime and always run. To exercise everything locally:

    python -m pytest tests/models/test_onnxruntime_parity.py -vv
"""

import tempfile

import numpy as np
import pytest

from lm_eval.api.instance import Instance


# Summed log-probs over a continuation: both engines run the same fp32 CPU
# kernels, so agreement should be near-exact. Kept loose enough to absorb
# accumulation-order differences between a single prompt pass and GenAI's
# internal buffering.
LOGLIKELIHOOD_ATOL = 1e-3

PROMPT_PAIRS = [
    ("The capital of France is", " Paris"),
    ("The capital of France is", " London"),
    ("Water boils at", " 100 degrees"),
    ("Water boils at", " zero degrees"),
    ("She opened the door and", " walked inside"),
    ("She opened the door and", " the sky turned green"),
    ("One plus one equals", " two"),
    ("One plus one equals", " seven"),
    ("The largest planet is", " Jupiter"),
    ("The largest planet is", " Mercury"),
    ("Cats are", " mammals"),
    ("Cats are", " reptiles"),
    ("In the morning I drink", " coffee"),
    ("In the morning I drink", " concrete"),
    ("Python is a", " programming language"),
    ("Python is a", " kind of bread"),
    ("The sun rises in the", " east"),
    ("The sun rises in the", " refrigerator"),
    ("Hello", " world"),
    ("A triangle has", " three sides"),
]


def _build_tiny_model_dir():
    """Build a tiny random Llama Model Builder export (fp32, CPU).

    head_size must be a multiple of 16 for the genai CPU GroupQueryAttention
    kernel, so hidden_size=128 / num_heads=8 -> head_size=16.
    """
    builder = pytest.importorskip("onnxruntime_genai.models.builder")
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    src = tempfile.mkdtemp(prefix="ort_parity_src_")
    out = tempfile.mkdtemp(prefix="ort_parity_out_")
    cache = tempfile.mkdtemp(prefix="ort_parity_cache_")

    tokenizer = AutoTokenizer.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM"
    )
    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=512,
        tie_word_embeddings=True,
    )
    LlamaForCausalLM(config).save_pretrained(src)
    tokenizer.save_pretrained(src)

    try:
        builder.create_model(None, src, out, "fp32", "cpu", cache)
    except Exception as e:  # noqa: BLE001 - any builder failure means skip
        pytest.skip(f"Model Builder could not create a CPU fixture: {e}")
    return out


def _make_requests(pairs):
    return [
        Instance(request_type="loglikelihood", doc={}, arguments=pair, idx=i)
        for i, pair in enumerate(pairs)
    ]


# --------------------------------------------------------------------------- #
# Registration: cheap, no runtime or model needed.
# --------------------------------------------------------------------------- #
def test_raw_backend_registered():
    from lm_eval.api.registry import get_model

    assert get_model("onnxruntime").__name__ == "ONNXRuntimeLM"


def test_both_backends_share_one_base():
    """The parity guarantee rests on a single shared implementation."""
    from lm_eval.models._onnx_base import _ONNXLMBase
    from lm_eval.models.onnxruntime_genai import ONNXRuntimeGenAILM
    from lm_eval.models.onnxruntime_ort import ONNXRuntimeLM

    assert issubclass(ONNXRuntimeGenAILM, _ONNXLMBase)
    assert issubclass(ONNXRuntimeLM, _ONNXLMBase)
    for method in ("_loglikelihood_tokens", "loglikelihood_rolling"):
        assert getattr(ONNXRuntimeGenAILM, method) is getattr(_ONNXLMBase, method)
        assert getattr(ONNXRuntimeLM, method) is getattr(_ONNXLMBase, method)


# --------------------------------------------------------------------------- #
# Provider resolution: no runtime needed, so these run in CI.
# --------------------------------------------------------------------------- #
class _FakeORT:
    """Stands in for the ``onnxruntime`` module during provider resolution."""

    def __init__(self, available):
        self._available = available

    def get_available_providers(self):
        return list(self._available)


def _make_provider_lm(execution_provider, declared=None, provider_options=None):
    from lm_eval.models.onnxruntime_ort import ONNXRuntimeLM

    lm = ONNXRuntimeLM.__new__(ONNXRuntimeLM)
    lm.execution_provider = execution_provider
    lm.provider_options = provider_options or {}
    lm._genai_config = {
        "model": {"decoder": {"session_options": {"provider_options": declared or []}}}
    }
    return lm


def test_providers_default_honors_genai_config():
    # Regression (mirrors test_select_ep_none_leaves_config_untouched for the
    # genai backend): with no execution_provider, the providers Model Builder
    # declared must be used, else a CUDA/DML/NPU export silently runs on CPU.
    lm = _make_provider_lm(None, declared=[{"cuda": {"device_id": "0"}}])
    providers, options = lm._resolve_providers(
        _FakeORT(["CUDAExecutionProvider", "CPUExecutionProvider"])
    )
    assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert options[0] == {"device_id": "0"}


def test_providers_default_without_declaration_is_cpu():
    """An empty provider list means CPU in genai, so match that."""
    lm = _make_provider_lm(None, declared=[])
    providers, _ = lm._resolve_providers(_FakeORT(["CPUExecutionProvider"]))
    assert providers == ["CPUExecutionProvider"]


def test_providers_explicit_request_overrides_declaration():
    lm = _make_provider_lm("rocm", declared=[{"cuda": {}}])
    providers, _ = lm._resolve_providers(
        _FakeORT(
            ["ROCMExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        )
    )
    # CPU stays last as a per-node fallback for ops the EP lacks.
    assert providers == ["ROCMExecutionProvider", "CPUExecutionProvider"]


def test_providers_explicit_cpu_forces_cpu():
    lm = _make_provider_lm("cpu", declared=[{"cuda": {}}])
    providers, _ = lm._resolve_providers(
        _FakeORT(["CUDAExecutionProvider", "CPUExecutionProvider"])
    )
    assert providers == ["CPUExecutionProvider"]


def test_unavailable_provider_raises():
    lm = _make_provider_lm("cuda")
    with pytest.raises(ValueError, match="not available"):
        lm._resolve_providers(_FakeORT(["CPUExecutionProvider"]))


# --------------------------------------------------------------------------- #
# Parity: both engines, same export, same (CPU) execution provider.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tiny_model_dir():
    pytest.importorskip("onnxruntime_genai")
    pytest.importorskip("onnxruntime")
    return _build_tiny_model_dir()


@pytest.fixture(scope="module")
def backends(tiny_model_dir):
    from lm_eval.api.registry import get_model

    args = f"pretrained={tiny_model_dir},execution_provider=cpu"
    genai = get_model("onnxruntime-genai").create_from_arg_string(
        args, {"batch_size": 1}
    )
    raw = get_model("onnxruntime").create_from_arg_string(args, {"batch_size": 1})
    return genai, raw


def test_forward_logits_parity(backends):
    """The raw single-pass forward matches GenAI's prompt logits elementwise."""
    genai, raw = backends
    tokens = genai.tok_encode("The quick brown fox jumps over the lazy dog")
    assert len(tokens) > 1

    genai_logits = genai._forward_logits(tokens)
    raw_logits = raw._forward_logits(tokens)

    assert genai_logits.shape == raw_logits.shape
    np.testing.assert_allclose(genai_logits, raw_logits, rtol=1e-3, atol=1e-3)


def test_loglikelihood_parity(backends):
    """Summed continuation log-probs and greedy flags agree across engines."""
    genai, raw = backends
    requests = _make_requests(PROMPT_PAIRS)

    genai_results = genai.loglikelihood(requests, disable_tqdm=True)
    raw_results = raw.loglikelihood(requests, disable_tqdm=True)

    assert len(genai_results) == len(raw_results) == len(PROMPT_PAIRS)

    diffs = []
    for (context, continuation), (ll_g, greedy_g), (ll_r, greedy_r) in zip(
        PROMPT_PAIRS, genai_results, raw_results, strict=True
    ):
        assert np.isfinite(ll_g) and np.isfinite(ll_r)
        assert ll_g < 0.0, "a real continuation must have negative log-prob"
        diffs.append(abs(ll_g - ll_r))
        assert abs(ll_g - ll_r) < LOGLIKELIHOOD_ATOL, (
            f"log-prob mismatch for {context!r} + {continuation!r}: "
            f"genai={ll_g:.6f} raw={ll_r:.6f}"
        )
        # is_greedy drives multiple_choice scoring, so it must agree too.
        assert greedy_g == greedy_r, (
            f"is_greedy mismatch for {context!r} + {continuation!r}: "
            f"genai={greedy_g} raw={greedy_r}"
        )

    print(f"\nmax summed-logprob diff over {len(diffs)} requests: {max(diffs):.3e}")


def test_multiple_choice_argmax_parity(backends):
    """The argmax over choices -- i.e. the predicted answer -- is identical.

    This is what multiple_choice accuracy (mmlu/hellaswag/arc) reduces to, so
    agreement here is the property that makes task metrics comparable.
    """
    genai, raw = backends
    # Group the fixed pairs by context: each group is one multiple-choice item.
    groups: dict[str, list[tuple[str, str]]] = {}
    for context, continuation in PROMPT_PAIRS:
        groups.setdefault(context, []).append((context, continuation))
    groups = {ctx: pairs for ctx, pairs in groups.items() if len(pairs) > 1}
    assert groups, "need at least one multi-choice group"

    for context, pairs in groups.items():
        requests = _make_requests(pairs)
        genai_lls = [ll for ll, _ in genai.loglikelihood(requests, disable_tqdm=True)]
        raw_lls = [ll for ll, _ in raw.loglikelihood(requests, disable_tqdm=True)]
        assert int(np.argmax(genai_lls)) == int(np.argmax(raw_lls)), (
            f"different answer chosen for {context!r}: genai={genai_lls} raw={raw_lls}"
        )


def test_rolling_perplexity_parity(backends):
    """loglikelihood_rolling (wikitext-style perplexity) agrees across engines."""
    genai, raw = backends
    text = (
        "The history of computing is long and winding. Early machines were "
        "mechanical, then electromechanical, and finally electronic. Each step "
        "made computation cheaper and more widely available, which in turn "
        "changed what people expected a computer to be able to do."
    )
    requests = [
        Instance(request_type="loglikelihood_rolling", doc={}, arguments=(text,), idx=0)
    ]

    (genai_ll,) = genai.loglikelihood_rolling(requests, disable_tqdm=True)
    (raw_ll,) = raw.loglikelihood_rolling(requests, disable_tqdm=True)

    assert np.isfinite(genai_ll) and np.isfinite(raw_ll)
    # Rolling sums accumulate over windows, so scale tolerance with magnitude.
    np.testing.assert_allclose(genai_ll, raw_ll, rtol=1e-4, atol=LOGLIKELIHOOD_ATOL)


def test_raw_backend_reports_graph_contract(backends):
    """The raw backend introspects the graph instead of assuming its inputs."""
    _, raw = backends
    input_names = {name for name, _, _ in raw._input_specs}
    assert "input_ids" in input_names
    assert "attention_mask" in input_names
    assert any(name.startswith("past_key_values") for name in input_names)
    assert raw._logits_name == "logits"
    # position_ids is absent from GQA+rotary exports; either way the backend
    # must agree with the graph rather than hardcode it.
    assert raw._has_position_ids == ("position_ids" in input_names)


def test_generative_tasks_rejected_clearly(backends):
    """Until the decode loop lands, generate_until must fail with guidance."""
    _, raw = backends
    requests = [
        Instance(
            request_type="generate_until",
            doc={},
            arguments=("Question: 2+2?\nAnswer:", {"until": ["\n"]}),
            idx=0,
        )
    ]
    with pytest.raises(NotImplementedError, match="onnxruntime-genai"):
        raw.generate_until(requests, disable_tqdm=True)
