from lm_eval.utils import escaped_split


def test_escaped_split_treats_regex_metacharacters_as_literals():
    assert escaped_split("a.b.c", ".") == ["a", "b", "c"]
    assert escaped_split("a|b|c", "|") == ["a", "b", "c"]
    assert escaped_split("a*b*c", "*") == ["a", "b", "c"]


def test_escaped_split_preserves_escaped_separator():
    assert escaped_split(r"a\.b.c", ".") == [r"a\.b", "c"]


def test_escaped_split_with_zero_maxsplit_returns_list():
    assert escaped_split("a,b", ",", maxsplit=0) == ["a,b"]


def test_escaped_split_honors_positive_maxsplit():
    assert escaped_split("a,b,c", ",", maxsplit=1) == ["a", "b,c"]
