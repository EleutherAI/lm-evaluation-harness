"""Tests for the context/continuation split in ``TemplateLM._encode_pair``.

For causal models the pair is tokenized joined and the result cut at
``len(tok_encode(context))``. That cut assumes no token spans the join. When
one does -- reported for extended and retrained vocabularies, where an entry
such as ``": C"`` covers the end of the context and the whole of a short
continuation -- the cut leaves the continuation with no tokens and evaluation
dies part-way through inference on ``assert len(continuation_enc) > 0``.

See #1053, #1297 and #3336.
"""

from unittest.mock import Mock

import pytest

from lm_eval.api.model import TemplateLM


# --------------------------------------------------------------------------
# A toy tokenizer: greedy longest match over a fixed vocabulary, falling back
# to one id per character. The entry that matters is ``": C"``, which is the
# vocabulary shape reported in #1053; the rest is scaffolding so that ordinary
# pairs tokenize the way a real BPE vocabulary would.
# --------------------------------------------------------------------------

_VOCAB = [": C", "Answer", "Question", " the", " C", "at", ":", " "]
_IDS = {piece: 1000 + i for i, piece in enumerate(_VOCAB)}
_LONGEST_FIRST = sorted(_VOCAB, key=len, reverse=True)


def toy_encode(string: str, add_special_tokens=None, **kwargs) -> list[int]:
    ids: list[int] = []
    i = 0
    while i < len(string):
        for piece in _LONGEST_FIRST:
            if string.startswith(piece, i):
                ids.append(_IDS[piece])
                i += len(piece)
                break
        else:
            ids.append(ord(string[i]))
            i += 1
    return ids


def make_lm(backend: str = "causal"):
    lm = Mock()
    lm.backend = backend
    lm.tok_encode = toy_encode
    lm._encode_pair = TemplateLM._encode_pair.__get__(lm, TemplateLM)
    return lm


# --------------------------------------------------------------------------


def test_toy_vocabulary_really_merges_across_the_join():
    """Guard the premise of the tests below, so they cannot pass vacuously."""
    assert len(toy_encode("Answer: C")) == len(toy_encode("Answer:"))


@pytest.mark.parametrize(
    "context, continuation",
    [
        ("Answer:", " C"),
        # the same pair as a task actually produces it: target_delimiter is
        # " " by default, so the space arrives on the end of the context and
        # _encode_pair moves it over before the split
        ("Answer: ", "C"),
    ],
)
def test_merged_boundary_leaves_something_to_score(context, continuation):
    context_enc, continuation_enc = make_lm()._encode_pair(context, continuation)

    assert continuation_enc, "continuation must not come back empty"
    assert continuation_enc == toy_encode(" C", add_special_tokens=False)
    assert context_enc == toy_encode("Answer:")


def test_exact_split_is_untouched():
    """The common case must produce byte-for-byte the tokens it always has."""
    lm = make_lm()
    context, continuation = "Question", " the"

    whole_enc = toy_encode(context + continuation)
    context_enc, continuation_enc = lm._encode_pair(context, continuation)

    assert context_enc == toy_encode(context)
    assert continuation_enc == whole_enc[len(context_enc) :]
    assert continuation_enc == toy_encode(" the")


def test_partial_merge_is_left_as_it_is():
    """A token spanning the join but leaving tokens behind is out of scope.

    ``"Answer: Cat"`` tokenizes as ``"Answer" | ": C" | "at"``, so the cut
    leaves ``"at"`` rather than ``" Cat"``. Where the probability mass of the
    spanning token belongs is a question about published numbers rather than a
    crash, and this test records that the empty-continuation fallback does not
    quietly answer it.
    """
    lm = make_lm()
    whole_enc = toy_encode("Answer: Cat")

    context_enc, continuation_enc = lm._encode_pair("Answer:", " Cat")

    assert continuation_enc == whole_enc[len(context_enc) :]
    assert continuation_enc == [_IDS["at"]]


def test_seq2seq_backend_encodes_separately_as_before():
    lm = make_lm(backend="seq2seq")

    context_enc, continuation_enc = lm._encode_pair("Answer:", " C")

    assert context_enc == toy_encode("Answer:")
    assert continuation_enc == toy_encode(" C", add_special_tokens=False)
