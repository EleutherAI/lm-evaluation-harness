import pytest

from lm_eval.filters.selection import TakeKFilter


@pytest.mark.parametrize("kwargs", [{}, {"k": 0}, {"k": -1}, {"k": "2"}, {"k": True}])
def test_take_first_k_rejects_invalid_k(kwargs):
    with pytest.raises(ValueError, match="requires a positive integer"):
        TakeKFilter(**kwargs)


def test_take_first_k_keeps_valid_behaviour():
    assert list(TakeKFilter(k=2).apply([["a", "b", "c"]], [{}])) == [["a", "b"]]
