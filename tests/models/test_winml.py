"""Tests for the winml backend.

``winml`` (:class:`lm_eval.models.winml.WindowsML`) is a thin subclass of
:class:`lm_eval.models.onnxruntime_genai.ONNXRuntimeGenAILM` that overrides only
execution-provider selection. The shared scoring/generation logic is covered by
``test_onnxruntime_genai.py``; this module covers the winml-specific wiring:
registration, the subclass relationship, and that Windows-provider registration
degrades gracefully to the cross-platform CPU path when the Windows ML APIs are
unavailable (e.g. off-Windows), so the inherited scoring core still runs.

Like the genai tests, these are excluded from CI (see the ``--ignore`` in
``.github/workflows/unit_tests.yml``); run locally with:

    python -m pytest tests/models/test_winml.py -vv
"""

import sys
import tempfile

import numpy as np
import pytest

from lm_eval.models.onnxruntime_genai import ONNXRuntimeGenAILM


pytest.importorskip("onnxruntime_genai")


def test_winml_registered_and_is_subclass():
    from lm_eval.api.registry import get_model
    from lm_eval.models.winml import WindowsML

    assert get_model("winml") is WindowsML
    assert issubclass(WindowsML, ONNXRuntimeGenAILM)
    # winml should specialize only provider selection + defaults, not the
    # scoring core.
    assert "_select_ep" in vars(WindowsML)
    for inherited in ("_loglikelihood_tokens", "loglikelihood", "generate_until"):
        assert inherited not in vars(WindowsML)


def test_winml_pins_historical_defaults(tiny_model_dir):
    # winml keeps its historical max_length / max_gen_toks so existing runs are
    # numerically unchanged, independent of the base backend's modern defaults.
    from lm_eval.api.registry import get_model

    lm = get_model("winml").create_from_arg_string(
        f"pretrained={tiny_model_dir}", {"batch_size": 1}
    )
    assert lm.max_length == 4096
    assert lm.max_gen_toks == 4096
    # explicit overrides still win
    lm2 = get_model("winml").create_from_arg_string(
        f"pretrained={tiny_model_dir},max_length=1024", {"batch_size": 1}
    )
    assert lm2.max_length == 1024


def _build_tiny_model_dir():
    """Build a tiny random Llama Model Builder export (fp32, CPU).

    head_size must be a multiple of 16 for the genai CPU GroupQueryAttention
    kernel, so hidden_size=128 / num_heads=8 -> head_size=16.
    """
    builder = pytest.importorskip("onnxruntime_genai.models.builder")
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    src = tempfile.mkdtemp(prefix="winml_src_")
    out = tempfile.mkdtemp(prefix="winml_out_")
    cache = tempfile.mkdtemp(prefix="winml_cache_")

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


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Exercises the non-Windows fallback path; on Windows the Windows ML "
    "catalog drives provider selection and requires a real NPU/GPU device.",
)
def test_winml_falls_back_to_cpu_off_windows(tiny_model_dir):
    """Off-Windows, winml should load via the base CPU path and score correctly."""
    from lm_eval.api.registry import get_model

    lm = get_model("winml").create_from_arg_string(
        f"pretrained={tiny_model_dir},execution_provider=cpu",
        {"batch_size": 1},
    )
    from lm_eval import evaluator

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=["hellaswag"],
        limit=2,
        num_fewshot=0,
        bootstrap_iters=0,
    )
    metrics = results["results"]["hellaswag"]
    numeric = [v for v in metrics.values() if isinstance(v, (int, float))]
    assert numeric
    assert all(np.isfinite(v) for v in numeric)
