import numpy as np
import pytest

from lm_eval.api.metrics import exact_match_hf_evaluate


@pytest.mark.parametrize(
    ("prediction", "reference", "expected"),
    [
        ("İ", "i\u0307", 1.0),
        ("i\u0307", "İ", 1.0),
        ("İ", "i", 0.0),
        ("i", "İ", 0.0),
        ("İx", "i\u0307x", 1.0),
        ("i\u0307x", "İx", 1.0),
        ("İx", "i\u0307", 0.0),
        ("HELLO", "hello", 1.0),
        ("Straße", "STRASSE", 0.0),
        ("Ä", "ä", 1.0),
    ],
)
def test_exact_match_preserves_lowercase_expansion(prediction, reference, expected):
    assert exact_match_hf_evaluate([prediction], [reference], ignore_case=True) == {
        "exact_match": expected
    }


@pytest.mark.parametrize("ignore_case", [False, True])
@pytest.mark.parametrize("ignore_punctuation", [False, True])
@pytest.mark.parametrize("ignore_numbers", [False, True])
@pytest.mark.parametrize("ignore_regex", [False, True])
def test_exact_match_case_expansion_preserves_normalization_order(
    ignore_case, ignore_punctuation, ignore_numbers, ignore_regex
):
    expected = float(
        ignore_case and ignore_punctuation and ignore_numbers and ignore_regex
    )
    assert exact_match_hf_evaluate(
        ["[DROP]İ1!a"],
        ["i\u03072?a"],
        regexes_to_ignore=[r"\[DROP\]"] if ignore_regex else None,
        ignore_case=ignore_case,
        ignore_punctuation=ignore_punctuation,
        ignore_numbers=ignore_numbers,
    ) == {"exact_match": expected}


@pytest.mark.parametrize(("reference", "expected"), [("i\u0307", 1.0), ("i", 0.0)])
def test_exact_match_case_expansion_does_not_depend_on_other_batch_items(
    reference, expected
):
    single = exact_match_hf_evaluate(["İ"], [reference], ignore_case=True)
    batch = exact_match_hf_evaluate(
        ["İ", "longer string"], [reference, "longer string"], ignore_case=True
    )
    assert single["exact_match"] == expected
    assert batch["exact_match"] == (expected + 1.0) / 2


@pytest.mark.parametrize("shape", [(), (1,), (1, 1)])
def test_exact_match_preserves_numpy_input_shape(shape):
    prediction = np.asarray("İ").reshape(shape)
    reference = np.asarray("i\u0307").reshape(shape)
    assert exact_match_hf_evaluate(prediction, reference, ignore_case=True) == {
        "exact_match": 1.0
    }


@pytest.mark.parametrize(("reference", "expected"), [("İ", 1.0), ("i\u0307", 0.0)])
def test_exact_match_keeps_case_sensitive_behavior(reference, expected):
    assert exact_match_hf_evaluate(["İ"], [reference]) == {"exact_match": expected}


def test_exact_match_keeps_byte_array_lowercase_behavior():
    assert exact_match_hf_evaluate(
        np.asarray([b"A"]), np.asarray([b"a"]), ignore_case=True
    ) == {"exact_match": 1.0}
    assert exact_match_hf_evaluate(
        np.asarray(["A"]), np.asarray([b"a"]), ignore_case=True
    ) == {"exact_match": 0.0}


@pytest.mark.parametrize("values", [np.asarray([1]), np.asarray(["A"], dtype=object)])
def test_exact_match_keeps_unsupported_lowercase_dtypes(values):
    with pytest.raises(TypeError):
        exact_match_hf_evaluate(values, values, ignore_case=True)


def test_exact_match_keeps_empty_unicode_array_behavior():
    values = np.asarray([], dtype="U1")
    with pytest.warns(RuntimeWarning):
        result = exact_match_hf_evaluate(values, values, ignore_case=True)
    assert np.isnan(result["exact_match"])
