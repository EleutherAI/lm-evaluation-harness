import os
from collections import defaultdict

from lm_eval.decontamination.janitor import (
    Janitor,
    form_ngrams,
    split_indices,
    word_ngrams,
    word_ngrams_indices,
)


os.environ["TOKENIZERS_PARALLELISM"] = "false"
TEST_SEQUENCE = (
    "Hello my name is Bob, I like eating pizza, chicken, chips and ice cream. Maybe I should eat some"
    " more salad but it's so booooring. I just... like eating pizza, chicken, chips and ice cream so much."
)

JANITOR_EXPECTED = (
    "This is a @line #containing a certain number of characters, 76 to be exact. "
    "This is a @line #containing a certain number of characters, 76 to be exact. "
    "This is a @line #containing a certain number of characters, 76 to be exact. "
    "This is a @line #containing "
    " characters, 76 to be exact. "
    "This is a @line #containing a certain number of characters, 76 to be exact. "
    "This is a @line #containing a certain number of characters, 76 to be exact. "
    "This is a @line #containing a certain number of characters, 76 to be exact. "
)

JANITOR_FILTH1 = "filth lots of dirty filthy filth"
JANITOR_FILTH2 = "filth lots of filthy dirty filth"


def simple_ngram(sequence, n):
    ngrams = list()
    ngram = []
    for x in sequence:
        ngram.extend([x])
        if len(ngram) == n:
            ngrams.extend([tuple(ngram)])
            ngram = ngram[1:]

    return ngrams


def test_form_ngrams():
    sequence = TEST_SEQUENCE

    n_values = [1, 2, 3, 5, 13]
    for n in n_values:
        comparison = simple_ngram(sequence, n)
        result_to_test = list(form_ngrams(iter(sequence), n))
        assert len(comparison) == len(result_to_test)
        assert comparison == result_to_test


def test_word_ngrams():
    sequence = TEST_SEQUENCE

    words = sequence.split()

    n_values = [1, 2, 3, 5, 13]
    for n in n_values:
        comparison = simple_ngram(words, n)
        comparison = [" ".join(ngram) for ngram in comparison]
        result_to_test = list(word_ngrams(sequence, n))
        assert len(comparison) == len(result_to_test)
        assert result_to_test == comparison


def test_split_indices():
    sequence = TEST_SEQUENCE

    comparison = []
    current_word = ""
    for i, c in enumerate(sequence):
        if c != " ":
            current_word += c
        else:
            if current_word:
                comparison.extend([(current_word, (i - len(current_word), i - 1))])
                current_word = ""

    if current_word:
        len_sequence = len(sequence)
        comparison.extend(
            [
                (
                    current_word,
                    (len_sequence - len(current_word), len_sequence - 1),
                )
            ]
        )
        current_word = ""

    result_to_test = list(split_indices(sequence))
    assert len(comparison) == len(result_to_test)
    assert comparison == result_to_test


def test_word_ngrams_indices():
    sequence = TEST_SEQUENCE

    n_values = [1, 2, 3, 5, 13]

    for n in n_values:
        ngrams = [" ".join(ngram) for ngram in simple_ngram(sequence.split(), n)]
        tracker = defaultdict(int)
        comparison = []
        for ngram in ngrams:
            while True:
                start = sequence.find(ngram, tracker[ngram])
                assert start != -1  # testing the test

                end = start + len(ngram) - 1
                tracker[ngram] = end + 1

                # ignore partial word matches
                if not (
                    (start != 0 and sequence[start - 1] != " ")
                    or (end != len(sequence) - 1 and sequence[end + 1] != " ")
                ):
                    break

            comparison.extend([(ngram, (start, end))])

        result_to_test = list(word_ngrams_indices(sequence, n))
        assert len(result_to_test) == len(comparison)
        assert result_to_test == comparison


# Assumptions from GPT3 Paper:
# the 200 characters to remove include punctuation and is actually a half-window


# All tests below initially test without any registered contaminants, expecting the same sequence back.
def test_janitor1():
    # First test using a 1gram and expected the first block before the filth to have some remaining
    # characters, but the second block should be completely removed.

    sequence = (
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "FILTH. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
    )

    filth = "filth"

    expected_result = (
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing "
    )

    janitor = Janitor(
        ngram_n=1, window_to_remove=200, too_dirty_cutoff=10, minimum_slice_length=200
    )
    result = janitor.clean_python(sequence)
    result = "".join(result)
    assert result == sequence

    janitor.register_contaminant(filth)
    assert janitor.dirt_ngrams == {filth}

    result = janitor.clean_python(sequence)
    result = "".join(result)
    assert result == expected_result


def test_janitor2():
    # Second test using a 1gram and expected the first block before the filth to have some remaining
    # characters, and the second block is longer then 200 characters so should also have some remaining.

    sequence = (
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "FILTH. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
    )

    filth = "filth"

    janitor = Janitor(
        ngram_n=1, window_to_remove=200, too_dirty_cutoff=10, minimum_slice_length=200
    )
    result = janitor.clean_python(sequence)
    result = "".join(result)
    assert result == sequence

    janitor.register_contaminant(filth)
    assert janitor.dirt_ngrams == {filth}

    result = janitor.clean_python(sequence)
    result = "".join(result)
    assert result == JANITOR_EXPECTED


def test_janitor3():
    # Same test as above but with a 6gram.

    sequence = (
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "FILTH. lots of dirty filtHy FIlTh "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
    )

    janitor = Janitor(
        ngram_n=6, window_to_remove=200, too_dirty_cutoff=10, minimum_slice_length=200
    )
    result = janitor.clean_python(sequence)
    result = "".join(result)
    assert result == sequence

    janitor.register_contaminant(JANITOR_FILTH1)
    assert janitor.dirt_ngrams == {JANITOR_FILTH1}

    result = janitor.clean_python(sequence)
    result = "".join(result)
    assert result == JANITOR_EXPECTED


def test_janitor4():
    # This test adds another block to that from the previous. The middle block should be entirely
    # removed as the 200 characters are removed from each side.

    sequence = (
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "FILTH. lots of dirty filtHy FIlTh "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "FILTH. lots of dirty filtHy FIlTh "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
    )

    janitor = Janitor(
        ngram_n=6, window_to_remove=200, too_dirty_cutoff=10, minimum_slice_length=200
    )
    result = janitor.clean_python(sequence)
    result = "".join(result)
    assert result == sequence

    janitor.register_contaminant(JANITOR_FILTH1)
    assert janitor.dirt_ngrams == {JANITOR_FILTH1}

    result = janitor.clean_python(sequence)
    result = "".join(result)
    assert result == JANITOR_EXPECTED


def test_janitor5():
    # Same as above but using multiple different filth 6grams.

    sequence = (
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "FILTH. lots of dirty filtHy FIlTh "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "FILTH. lots of filtHy dirty FIlTh "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
    )

    filths = [JANITOR_FILTH1, JANITOR_FILTH2]

    janitor = Janitor(
        ngram_n=6, window_to_remove=200, too_dirty_cutoff=10, minimum_slice_length=200
    )
    result = janitor.clean_python(sequence)
    result = "".join(result)
    assert result == sequence

    for filth in filths:
        janitor.register_contaminant(filth)
    assert janitor.dirt_ngrams == set(filths)

    result = janitor.clean_python(sequence)
    result = "".join(result)
    assert result == JANITOR_EXPECTED


def test_janitor6():
    # Same as above but now we add 10 filths and expect the same result, the following test does 11.

    sequence = (
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "FILTH. lots of dirty filtHy FIlTh "
        "FILTH. lots of dirty filtHy FIlTh "
        "FILTH. lots of dirty filtHy FIlTh "
        "FILTH. lots of dirty filtHy FIlTh "
        "FILTH. lots of dirty filtHy FIlTh "
        "FILTH. lots of dirty filtHy FIlTh "
        "FILTH. lots of dirty filtHy FIlTh "
        "FILTH. lots of dirty filtHy FIlTh "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "FILTH. lots of filtHy dirty FIlTh "
        "FILTH. lots of filtHy dirty FIlTh "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
    )

    filths = [JANITOR_FILTH1, JANITOR_FILTH2]

    janitor = Janitor(
        ngram_n=6, window_to_remove=200, too_dirty_cutoff=10, minimum_slice_length=200
    )
    result = janitor.clean_python(sequence)
    result = "".join(result)
    assert result == sequence

    for filth in filths:
        janitor.register_contaminant(filth)
    assert janitor.dirt_ngrams == set(filths)

    result = janitor.clean_python(sequence)
    result = "".join(result)
    assert result == JANITOR_EXPECTED


def test_janitor7():
    # Same as above but now we add 9 filths and expect the same result, the following test does 10.

    sequence = (
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "FILTH. lots of dirty filtHy FIlTh "
        "FILTH. lots of dirty filtHy FIlTh "
        "FILTH. lots of dirty filtHy FIlTh "
        "FILTH. lots of dirty filtHy FIlTh "
        "FILTH. lots of dirty filtHy FIlTh "
        "FILTH. lots of dirty filtHy FIlTh "
        "FILTH. lots of dirty filtHy FIlTh "
        "FILTH. lots of dirty filtHy FIlTh "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "FILTH. lots of filtHy dirty FIlTh "
        "FILTH. lots of filtHy dirty FIlTh "
        "FILTH. lots of filtHy dirty FIlTh "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
        "This is a @line #containing a certain number of characters, 76 to be exact. "
    )

    filths = [JANITOR_FILTH1, JANITOR_FILTH2]

    expected_result = ""

    janitor = Janitor(
        ngram_n=6, window_to_remove=200, too_dirty_cutoff=10, minimum_slice_length=200
    )
    result = janitor.clean_python(sequence)
    result = "".join(result)
    assert result == sequence

    for filth in filths:
        janitor.register_contaminant(filth)
    assert janitor.dirt_ngrams == set(filths)

    result = janitor.clean_python(sequence)
    result = "".join(result)
    assert result == expected_result


def test_janitor8():
    # This will test the save and load contams
    pass
