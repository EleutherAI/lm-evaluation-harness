import pytest

from lm_eval.filters.selection import TakeKFilter


def test_take_k_accepts_empty_response_batches():
    assert list(TakeKFilter(k=2).apply([], [])) == []


@pytest.mark.parametrize("use_iterator", [False, True])
def test_take_k_validates_later_documents(use_iterator):
    resps = [["a", "b"], ["c"]]
    if use_iterator:
        resps = iter(resps)
    with pytest.raises(AssertionError, match="got 1 only"):
        list(TakeKFilter(k=2).apply(resps, [{}, {}]))


def test_take_k_preserves_document_and_response_order():
    resps = iter([["a", "b", "c"], ["d", "e"]])
    assert list(TakeKFilter(k=2).apply(resps, [{}, {}])) == [["a", "b"], ["d", "e"]]


def test_take_k_still_rejects_short_first_document():
    with pytest.raises(AssertionError, match="got 1 only"):
        list(TakeKFilter(k=2).apply([["a"], ["b", "c"]], [{}, {}]))
