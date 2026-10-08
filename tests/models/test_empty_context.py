"""Tests for the empty-context contract of ``TemplateLM.loglikelihood``.

``LM.loglikelihood`` documents that *implementations must handle empty string*
contexts (``lm_eval/api/model.py``), and ``_encode_pair`` documents that the
caller owns that case. ``loglikelihood`` guarded the literal ``""`` but not a
context that becomes empty one call deeper: ``_encode_pair`` migrates trailing
whitespace into the continuation, so an all-whitespace context is left with
nothing to encode, and tokenizers that map ``""`` to no tokens (GPT-2, Qwen2,
Mistral, Llama-3) then produce an empty ``context_enc``. The scoring backends
reject that on ``assert len(context_enc) > 0``, losing the whole run.

``MegatronLMEval.loglikelihood`` already carries the guarded form of this same
algorithm; this pins ``TemplateLM`` to the same behaviour.

A toy tokenizer stands in for a real one so the tests stay hermetic: it maps
``""`` to no tokens, as the tokenizers named above do.
"""

from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM


BOS = 1


class ToyLM(TemplateLM):
    """A ``TemplateLM`` with just enough of a tokenizer to drive the encoding.

    Character-per-token, with two properties that matter here: ``""`` encodes to
    no tokens, and a tokenizer that prepends BOS can be simulated by handing
    the continuation in already-BOS-prefixed.
    """

    def __init__(self, backend="causal", prefix_token_id=BOS):
        self.backend = backend
        self._prefix_token_id = prefix_token_id

    @property
    def prefix_token_id(self):
        return self._prefix_token_id

    def tok_encode(self, string, add_special_tokens=None, **kwargs):
        # Real causal tokenizers map "" to an empty id list.
        if string == "":
            return []
        return [ord(char) for char in string]

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        # Echo the encoded triples so tests can assert on what would be scored.
        return [
            ((context, continuation), context_enc, continuation_enc)
            for (context, continuation), context_enc, continuation_enc in requests
        ]

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests):
        raise NotImplementedError

    @property
    def eot_token_id(self):
        return 2


def _request(context, continuation):
    return Instance(
        request_type="loglikelihood",
        doc={},
        arguments=(context, continuation),
        idx=0,
    )


def _encode(lm, context, continuation):
    """Return the (context_enc, continuation_enc) pair a request would be scored with."""
    ((_, _), context_enc, continuation_enc) = lm.loglikelihood(
        [_request(context, continuation)]
    )[0]
    return context_enc, continuation_enc


class TestEmptyContextContract:
    def test_literal_empty_context_uses_prefix_token(self):
        """The documented case: an empty context conditions on the prefix token."""
        assert _encode(ToyLM(), "", " Paris") == ([BOS], list(b" Paris"))


class TestWhitespaceOnlyContext:
    """A context of only whitespace is empty once trailing spaces are migrated."""

    def test_single_space_context_is_not_left_empty(self):
        # The space migrates to the continuation, so the context has nothing
        # left; it must fall back to the prefix token rather than encode to [].
        context_enc, continuation_enc = _encode(ToyLM(), " ", " Paris")

        assert context_enc == [BOS]
        # The migrated space is kept, so the scored text is unchanged.
        assert continuation_enc == list(b"  Paris")

    def test_newline_context_is_not_left_empty(self):
        context_enc, _ = _encode(ToyLM(), "\n", " Paris")

        assert context_enc == [BOS]

    def test_mixed_whitespace_context_is_not_left_empty(self):
        context_enc, _ = _encode(ToyLM(), "  \t ", " Paris")

        assert context_enc == [BOS]

    def test_ordinary_context_is_unchanged(self):
        """The fallback must not disturb a context that has real content."""
        context_enc, continuation_enc = _encode(ToyLM(), "hello", " world")

        assert context_enc == list(b"hello")
        assert continuation_enc == list(b" world")

    def test_trailing_space_on_real_context_still_migrates(self):
        """Word-boundary tokenization is preserved: the split is unchanged."""
        context_enc, continuation_enc = _encode(ToyLM(), "x ", "ab")

        assert context_enc == list(b"x")
        assert continuation_enc == list(b" ab")
