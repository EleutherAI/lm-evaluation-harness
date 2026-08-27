"""Tests for the cross-platform onnxruntime-genai backend.

The heavy smoke tests require ``onnxruntime-genai`` plus its Model Builder and
build a tiny random Llama model on the fly (CPU execution provider), mirroring
the export-on-the-fly pattern in ``tests/models/test_openvino.py``. They are
excluded from CI (see the ``--ignore`` in ``.github/workflows/unit_tests.yml``)
because the ONNX runtimes are not installed there; run them locally with:

    python -m pytest tests/models/test_onnxruntime_genai.py -vv
"""

import tempfile

import numpy as np
import pytest

from lm_eval.api.model import CacheHook
from lm_eval.models.onnxruntime_genai import ONNXRuntimeGenAILM, _log_softmax


pytest.importorskip("onnxruntime_genai")


def _build_tiny_model_dir():
    """Build a tiny random Llama Model Builder export (fp32, CPU).

    head_size must be a multiple of 16 for the genai CPU GroupQueryAttention
    kernel, so hidden_size=128 / num_heads=8 -> head_size=16.
    """
    builder = pytest.importorskip("onnxruntime_genai.models.builder")
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    src = tempfile.mkdtemp(prefix="ort_genai_src_")
    out = tempfile.mkdtemp(prefix="ort_genai_out_")
    cache = tempfile.mkdtemp(prefix="ort_genai_cache_")

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
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Model Builder could not create a CPU fixture: {e}")
    return out


@pytest.fixture(scope="session")
def tiny_model_dir():
    return _build_tiny_model_dir()


# --------------------------------------------------------------------------- #
# Unit tests: scoring math, no model load required.
# --------------------------------------------------------------------------- #
def _make_bare_lm(forward_logits, max_length=2048):
    """Build an instance without loading a model, wiring only what the token scoring path needs."""
    lm = ONNXRuntimeGenAILM.__new__(ONNXRuntimeGenAILM)
    lm.max_length = max_length
    lm.cache_hook = CacheHook(None)
    lm._forward_logits = forward_logits
    return lm


def test_log_softmax_matches_reference():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(4, 7)).astype(np.float32)
    got = _log_softmax(logits)
    # Reference: log(exp(x) / sum(exp(x)))
    ref = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)
    # Each row is a valid log-prob distribution (sums to 1 in prob space).
    np.testing.assert_allclose(np.exp(got).sum(axis=-1), np.ones(4), atol=1e-5)


def test_loglikelihood_tokens_math():
    # vocab=5; craft logits so the continuation tokens are the argmax everywhere.
    vocab = 5
    context_enc = [1, 2]
    continuation_enc = [3, 4]
    full_len = len(context_enc) + len(continuation_enc)

    # Deterministic logits: position i strongly predicts continuation target.
    logits = np.full((full_len, vocab), -10.0, dtype=np.float32)
    # Row (ctxlen-1 + j) should predict continuation_enc[j].
    ctxlen = len(context_enc)
    for j, tok in enumerate(continuation_enc):
        logits[ctxlen - 1 + j, tok] = 10.0

    lm = _make_bare_lm(lambda toks: logits)
    ((ll, is_greedy),) = lm._loglikelihood_tokens(
        [(None, context_enc, continuation_enc)], disable_tqdm=True
    )

    # Manually compute expected loglikelihood from the same logits.
    lp = _log_softmax(logits[ctxlen - 1 : ctxlen - 1 + len(continuation_enc)])
    expected = float(lp[np.arange(len(continuation_enc)), continuation_enc].sum())
    assert ll == pytest.approx(expected)
    assert is_greedy is True


def test_loglikelihood_tokens_not_greedy():
    vocab = 5
    context_enc = [1]
    continuation_enc = [2]
    # Predict a *different* token as argmax, so is_greedy must be False.
    logits = np.full((2, vocab), -10.0, dtype=np.float32)
    logits[0, 4] = 10.0  # argmax is 4, but target is 2
    lm = _make_bare_lm(lambda toks: logits)
    ((ll, is_greedy),) = lm._loglikelihood_tokens(
        [(None, context_enc, continuation_enc)], disable_tqdm=True
    )
    assert is_greedy is False
    assert ll < 0.0


def test_loglikelihood_tokens_empty_continuation():
    lm = _make_bare_lm(lambda toks: np.zeros((1, 3), dtype=np.float32))
    (result,) = lm._loglikelihood_tokens([(None, [1], [])], disable_tqdm=True)
    assert result == (0.0, True)


class _FakeConfig:
    """Records provider mutations made by _select_ep."""

    def __init__(self):
        self.cleared = False
        self.appended = []
        self.options = []

    def clear_providers(self):
        self.cleared = True

    def append_provider(self, ep):
        self.appended.append(ep)

    def set_provider_option(self, ep, key, value):
        self.options.append((ep, key, value))


def _make_ep_lm(execution_provider, provider_options=None):
    lm = ONNXRuntimeGenAILM.__new__(ONNXRuntimeGenAILM)
    lm.execution_provider = execution_provider
    lm.provider_options = provider_options or {}
    return lm


def test_select_ep_none_leaves_config_untouched():
    # Regression: with no execution_provider, the providers Model Builder wrote
    # into genai_config.json must survive (clear_providers must NOT be called),
    # else a CUDA/DML/NPU export would be silently downgraded to CPU.
    cfg = _FakeConfig()
    _make_ep_lm(None)._select_ep(cfg)
    assert cfg.cleared is False
    assert cfg.appended == []


def test_select_ep_cpu_clears_without_appending():
    cfg = _FakeConfig()
    _make_ep_lm("cpu")._select_ep(cfg)
    assert cfg.cleared is True
    assert cfg.appended == []


def test_select_ep_named_provider_clears_and_appends():
    cfg = _FakeConfig()
    _make_ep_lm("cuda", {"device_id": 0})._select_ep(cfg)
    assert cfg.cleared is True
    assert cfg.appended == ["cuda"]
    assert cfg.options == [("cuda", "device_id", "0")]


# --------------------------------------------------------------------------- #
# Smoke tests: build a tiny model and run real tasks end-to-end on CPU.
# --------------------------------------------------------------------------- #
@pytest.fixture
def lm(tiny_model_dir):
    from lm_eval.api.registry import get_model

    return get_model("onnxruntime-genai").create_from_arg_string(
        f"pretrained={tiny_model_dir},execution_provider=cpu",
        {"batch_size": 1},
    )


def test_backend_loads(lm):
    assert lm.eot_token_id is not None
    assert lm.max_length > 0
    ids = lm.tok_encode("hello world")
    assert isinstance(ids, list) and len(ids) > 0
    # Tokenization round-trips (up to special-token stripping).
    assert "hello" in lm.tok_decode(ids)


@pytest.mark.parametrize("task", ["hellaswag", "wikitext", "gsm8k"])
def test_simple_evaluate(lm, task):
    from lm_eval import evaluator

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=[task],
        limit=2,
        num_fewshot=0,
        bootstrap_iters=0,
    )
    assert task in results["results"]
    metrics = results["results"][task]
    # Some real metric must be present and finite (no silent 0.0-on-error).
    numeric = [v for v in metrics.values() if isinstance(v, (int, float))]
    assert numeric, f"no numeric metrics for {task}: {metrics}"
    assert all(np.isfinite(v) for v in numeric)


def test_loglikelihood_deterministic(lm):
    """The same request scored twice yields identical log-probs (greedy path)."""
    from lm_eval.api.instance import Instance

    req = Instance(
        request_type="loglikelihood",
        doc={},
        arguments=("The capital of France is", " Paris"),
        idx=0,
    )
    (first,) = lm.loglikelihood([req], disable_tqdm=True)
    (second,) = lm.loglikelihood([req], disable_tqdm=True)
    assert first[0] == pytest.approx(second[0])
    assert first[0] < 0.0  # a real, non-zero log-prob
