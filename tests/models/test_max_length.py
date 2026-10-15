"""Unit tests for context-length auto-detection.

Regression coverage for multimodal configs (e.g. Gemma3) that nest the text
model's context length under ``text_config`` instead of exposing it at the top
level of ``model.config``. See EleutherAI/lm-evaluation-harness#3460.

The resolution logic is shared by every config-scanning backend via
``lm_eval.models.utils.resolve_max_length``; the tests below cover the resolver
directly and then pin each backend's ``max_length`` property to it. Lightweight
config stubs are used throughout, so no model weights are downloaded.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import transformers

from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils import TOKENIZER_INFINITY, resolve_max_length


DEFAULT = HFLM._DEFAULT_MAX_LENGTH


def _config(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


def _tokenizer(model_max_length) -> SimpleNamespace:
    return SimpleNamespace(model_max_length=model_max_length)


class TestResolveMaxLength:
    """Direct coverage of the shared resolver."""

    def test_reads_nested_text_config(self):
        """Gemma3-style configs nest the context length under ``text_config``.

        The attribute is absent at the top level, and the processor tokenizer
        reports an infinite ``model_max_length``, so without a nested lookup the
        context length silently truncates to ``_DEFAULT_MAX_LENGTH`` (2048).
        This is the reported bug.
        """
        config = _config(text_config=_config(max_position_embeddings=131072))
        assert (
            resolve_max_length(config, _tokenizer(TOKENIZER_INFINITY), default=DEFAULT)
            == 131072
        )

    def test_reads_top_level_config(self):
        """Text-only configs expose the attribute at the top level."""
        config = _config(max_position_embeddings=4096)
        assert resolve_max_length(config, _tokenizer(2048), default=DEFAULT) == 4096

    def test_nested_text_config_wins_over_top_level(self):
        """``text_config`` is preferred when a composite config has both.

        On composite configs the top-level attribute can belong to a modality
        encoder rather than the text model (``MusicFlamingoConfig`` reports 1200
        there while its text model handles 32768), so the nested value is the
        one that describes how many tokens the LM can accept.
        """
        config = _config(
            max_position_embeddings=1200,
            text_config=_config(max_position_embeddings=32768),
        )
        assert resolve_max_length(config, None, default=DEFAULT) == 32768

    def test_config_wins_over_smaller_tokenizer(self):
        """A config value is not capped by a smaller tokenizer value."""
        config = _config(text_config=_config(max_position_embeddings=32768))
        assert resolve_max_length(config, _tokenizer(4096), default=DEFAULT) == 32768

    def test_attr_precedence(self):
        """``n_positions`` is checked before ``max_position_embeddings``."""
        config = _config(n_positions=1024, max_position_embeddings=2048, n_ctx=512)
        assert resolve_max_length(config, None, default=DEFAULT) == 1024

    def test_skips_attrs_set_to_none(self):
        """An attribute present but set to ``None`` falls through to the next."""
        config = _config(n_positions=None, max_position_embeddings=4096)
        assert resolve_max_length(config, None, default=DEFAULT) == 4096

    def test_coerces_to_int(self):
        """A float context length is coerced, since callers index with it."""
        config = _config(max_position_embeddings=1200.0)
        resolved = resolve_max_length(config, None, default=DEFAULT)
        assert resolved == 1200
        assert isinstance(resolved, int)

    def test_falls_back_to_tokenizer(self):
        """With no config value, a finite tokenizer length is used."""
        assert resolve_max_length(_config(), _tokenizer(8192), default=DEFAULT) == 8192

    def test_ignores_tokenizer_infinity(self):
        """The tokenizer's "unset" sentinel is not a real context length."""
        assert (
            resolve_max_length(
                _config(), _tokenizer(TOKENIZER_INFINITY), default=DEFAULT
            )
            == DEFAULT
        )

    def test_falls_back_to_default(self):
        """No config and no tokenizer leaves only the default."""
        assert resolve_max_length(_config(), None, default=DEFAULT) == DEFAULT

    def test_tolerates_missing_config(self):
        """A backend may not have loaded a config at all."""
        assert resolve_max_length(None, _tokenizer(4096), default=DEFAULT) == 4096


class TestHFLMMaxLength:
    """Pin ``HFLM.max_length`` to the shared resolver.

    The real property getter is run against minimal stubs, so the wiring is
    covered without downloading model weights.
    """

    @staticmethod
    def _max_length(config, model_max_length):
        stub = SimpleNamespace(
            _max_length=None,
            _DEFAULT_MAX_LENGTH=DEFAULT,
            model=SimpleNamespace(config=config),
            tokenizer=_tokenizer(model_max_length),
        )
        return HFLM.max_length.fget(stub)

    def test_manual_override_wins(self):
        """An explicit ``max_length=`` argument short-circuits detection."""
        stub = SimpleNamespace(_max_length=512)
        assert HFLM.max_length.fget(stub) == 512

    def test_nested_text_config(self):
        """The reported bug: a nested value must not fall through to 2048."""
        config = _config(text_config=_config(max_position_embeddings=131072))
        assert self._max_length(config, TOKENIZER_INFINITY) == 131072

    def test_top_level_config(self):
        """Text-only configs keep resolving from the top level."""
        assert self._max_length(_config(max_position_embeddings=4096), 2048) == 4096

    def test_real_gemma3_config(self):
        """Pin the fix against a real ``Gemma3Config``.

        Guards against a future transformers change to where the context length
        nests, which a hand-built stub would not catch.
        """
        if not hasattr(transformers, "Gemma3Config"):
            pytest.skip("installed transformers has no Gemma3Config")
        config = transformers.Gemma3Config()
        expected = config.text_config.max_position_embeddings
        assert expected > DEFAULT
        assert self._max_length(config, TOKENIZER_INFINITY) == expected


def test_vllm_data_parallel_max_length():
    """Pin the vLLM data-parallel branch to the shared resolver.

    With ``data_parallel_size > 1`` there is no engine to query, so vLLM scans
    the config and hit the same nested-config bug as the HF backend.
    """
    pytest.importorskip("vllm")
    from lm_eval.models.vllm_causallms import VLLM

    stub = SimpleNamespace(
        _max_length=None,
        _DEFAULT_MAX_LENGTH=VLLM._DEFAULT_MAX_LENGTH,
        data_parallel_size=2,
        _config=_config(text_config=_config(max_position_embeddings=131072)),
        tokenizer=_tokenizer(TOKENIZER_INFINITY),
    )
    assert VLLM.max_length.fget(stub) == 131072
