"""Parity tests between the optional C++ janitor (janitor_util) and the
pure-Python reference implementation in lm_eval/decontamination/janitor.py.

The C++ module is not built in CI, so the whole file skips when it cannot be
imported. When it is available, C++ mode must produce exactly the ngrams,
character indices, and clean() output that Python mode produces, including
for non-ASCII text (regression coverage for #1452, where the C++ code
counted UTF-8 bytes instead of characters and could raise
UnicodeDecodeError on multibyte input).
"""

import pytest


janitor_util = pytest.importorskip("janitor_util")

from lm_eval.decontamination.janitor import (
    Janitor,
    word_ngrams,
    word_ngrams_indices,
)


DELETE_CHARS = Janitor().delete_chars
_normalize = Janitor().normalize_string

TEXTS = [
    "Hello my name is Bob, I like eating pizza and ice cream.",
    # multibyte words longer than the old 10-byte gram cap
    "überwältigend sauberer Text läuft weiter",
    # one word far longer than 10 characters/bytes
    "pneumonoultramicroscopicsilicovolcanoconiosis is long",
    # CJK (3-byte characters); first token has no internal spaces
    "日本語のテキストです テストデータ 汚染チェック",
    # 4-byte emoji sequences
    "emoji 😀 test 🎉 words here now",
    # accented Latin text
    "des mots en français avec des caractères accentués",
    # en/em dashes are in delete_chars and vanish from grams
    "a—dash–joined word stays",
    # no trailing whitespace: trailing ngrams must still be produced
    "trailing words without whitespace",
    "word",
    "",
    "   ",
]


@pytest.mark.parametrize("text", TEXTS)
@pytest.mark.parametrize("n", [1, 2, 3])
def test_clean_ngram_matches_python(text, n):
    expected = list(word_ngrams(_normalize(text), n))
    assert janitor_util.clean_ngram(text, DELETE_CHARS, n) == expected


@pytest.mark.parametrize("text", TEXTS)
@pytest.mark.parametrize("n", [1, 2])
def test_clean_ngram_with_indices_matches_python(text, n):
    expected = [
        (_normalize(gram), indices) for gram, indices in word_ngrams_indices(text, n)
    ]
    result = [
        (_normalize(gram), (start, end))
        for gram, start, end in janitor_util.clean_ngram_with_indices(
            text, DELETE_CHARS, n
        )
    ]
    assert result == expected


def test_clean_cpp_matches_python_on_multibyte_text():
    words = [f"wörter{i}ß" for i in range(30)]
    dirty = " ".join(words)
    contaminant = " ".join(words[10:23])

    jan_py = Janitor(ngram_n=13, window_to_remove=5, minimum_slice_length=3)
    jan_py.register_contaminant_python(contaminant)
    expected = jan_py.clean_python(dirty)

    jan_cpp = Janitor(ngram_n=13, window_to_remove=5, minimum_slice_length=3)
    jan_cpp.register_contaminant_cpp(contaminant)
    assert jan_cpp.clean_cpp(dirty) == expected


def test_clean_cpp_ignores_unregistered_ngrams():
    jan = Janitor(ngram_n=2, window_to_remove=5, minimum_slice_length=3)
    jan.register_contaminant_cpp("some registered contaminant text")
    text = "perfectly clean text that shares nothing with the contaminant"
    assert jan.clean_cpp(text) == jan.clean_python(text)
