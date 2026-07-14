"""Unit tests for ``HFLM.max_length`` context-length auto-detection.

Regression coverage for multimodal configs (e.g. Gemma3) that nest the text
model's context length under ``text_config`` instead of exposing it at the top
level of ``model.config``. See EleutherAI/lm-evaluation-harness#3460.

The tests exercise the real ``HFLM.max_length`` getter against lightweight
config stubs so no model weights are downloaded.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import transformers

from lm_eval.models.huggingface import HFLM, TOKENIZER_INFINITY


def _resolve_max_length(config, model_max_length):
    """Run the real ``HFLM.max_length`` getter against minimal stubs."""
    stub = SimpleNamespace(
        _max_length=None,
        _DEFAULT_MAX_LENGTH=HFLM._DEFAULT_MAX_LENGTH,
        model=SimpleNamespace(config=config),
        tokenizer=SimpleNamespace(model_max_length=model_max_length),
    )
    return HFLM.max_length.fget(stub)


def test_max_length_reads_nested_text_config():
    """Gemma3-style configs nest ``max_position_embeddings`` under
    ``text_config`` and omit it at the top level; the processor tokenizer
    reports an infinite ``model_max_length``. Without a nested fallback the
    context length silently truncates to ``_DEFAULT_MAX_LENGTH`` (2048).
    """
    config = SimpleNamespace(
        text_config=SimpleNamespace(max_position_embeddings=131072)
    )
    assert _resolve_max_length(config, TOKENIZER_INFINITY) == 131072


def test_max_length_top_level_config_takes_precedence():
    """Text-only configs expose the attribute at the top level and must keep
    resolving there, unaffected by the nested fallback.
    """
    config = SimpleNamespace(max_position_embeddings=4096)
    assert _resolve_max_length(config, 2048) == 4096


def test_max_length_reads_real_gemma3_config():
    """Pin the fix against a real ``Gemma3Config`` so a future transformers
    change to where the context length nests is caught, not only the stub.
    """
    if not hasattr(transformers, "Gemma3Config"):
        pytest.skip("installed transformers has no Gemma3Config")
    config = transformers.Gemma3Config()
    expected = config.text_config.max_position_embeddings
    assert expected > HFLM._DEFAULT_MAX_LENGTH
    assert _resolve_max_length(config, TOKENIZER_INFINITY) == expected


def test_max_length_nested_config_beats_finite_tokenizer():
    """A nested config value takes precedence over the tokenizer, mirroring
    the existing top-level behavior where a config value wins over a smaller
    tokenizer ``model_max_length`` rather than being capped by it.
    """
    config = SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=32768))
    assert _resolve_max_length(config, 4096) == 32768
