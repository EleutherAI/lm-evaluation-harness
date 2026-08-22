r"""Indexed-root survival through normalize_final_answer.

The sqrt shorthand rule used to consume the "[" of an optional root index,
turning e.g. \\sqrt[3]{8} into the invalid \\sqrt{[}3]{8}; see issue #4036.
"""

import pytest


minerva_utils = pytest.importorskip(
    "lm_eval.tasks.minerva_math.utils",
    reason="minerva_math needs the [math] extra: "
    "sympy, math_verify, antlr4-python3-runtime==4.11",
)

normalize = minerva_utils.normalize_final_answer
is_equiv = minerva_utils.is_equiv


@pytest.mark.parametrize(
    "answer",
    [
        "\\sqrt[3]{8}",
        "\\sqrt[3]{2}+1",
        "\\sqrt[4]{16}",
        # unbraced argument gets canonicalized to the braced form
        ("\\sqrt[3]8", "\\sqrt[3]{8}"),
        # classic shorthands must keep their old behavior byte-for-byte
        ("\\sqrt2", "\\sqrt{2}"),
        ("\\sqrta", "\\sqrt{a}"),
    ],
)
def test_normalize_preserves_indexed_roots(answer):
    expected = answer[1] if isinstance(answer, tuple) else answer
    assert normalize(answer[0] if isinstance(answer, tuple) else answer) == expected


@pytest.mark.parametrize(
    "pred,gold",
    [
        # model writes a cube root, gold is the simplified integer
        ("\\sqrt[3]{8}", "2"),
        # brace vs bare argument spellings of the same root
        ("\\sqrt[3]2", "\\sqrt[3]{2}"),
        # fourth power root vs its integer value
        ("\\sqrt[4]{16}", "2"),
    ],
)
def test_is_equiv_indexed_roots_through_pipeline(pred, gold):
    assert is_equiv(normalize(pred), normalize(gold))


def test_is_equiv_shorthand_still_works():
    assert is_equiv(normalize("\\sqrt8"), normalize("2\\sqrt{2}"))
