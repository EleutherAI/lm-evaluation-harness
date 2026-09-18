import pytest

from lm_eval._cli.utils import handle_cli_value_string, key_val_to_dict


@pytest.mark.parametrize(
    ("value", "expected", "expected_type"),
    [
        ("42", 42, int),
        ("-1", -1, int),
        ("+2", 2, int),
        ("-0.5", -0.5, float),
        ("1e3", 1000.0, float),
        ('"123"', "123", str),
        ("[1, -2]", [1, -2], list),
        ('{"limit": -1}', {"limit": -1}, dict),
    ],
)
def test_handle_cli_value_string_preserves_value_types(value, expected, expected_type):
    result = handle_cli_value_string(value)

    assert result == expected
    assert type(result) is expected_type


def test_key_val_to_dict_distinguishes_signed_integers_from_floats():
    result = key_val_to_dict("top_k=-1,max_tokens=+2,temperature=-0.5")

    assert result == {"top_k": -1, "max_tokens": 2, "temperature": -0.5}
    assert type(result["top_k"]) is int
    assert type(result["max_tokens"]) is int
    assert type(result["temperature"]) is float


def test_key_val_to_dict_strips_whitespace_from_keys():
    """Test that keys are stripped of surrounding whitespace (issue #4147)."""
    result = key_val_to_dict("do_sample=false, temperature=1.0")

    assert result == {"do_sample": False, "temperature": 1.0}
    assert " temperature" not in result


def test_key_val_to_dict_strips_multiple_spaces_from_keys():
    """Test that multiple spaces around keys are stripped."""
    result = key_val_to_dict("a=1,  b=2,   c=3")

    assert result == {"a": 1, "b": 2, "c": 3}


def test_key_val_to_dict_strips_tab_whitespace_from_keys():
    """Test that tab whitespace around keys is stripped."""
    result = key_val_to_dict("a=1,\tb=2")

    assert result == {"a": 1, "b": 2}


def test_key_val_to_dict_rejects_empty_key():
    """Test that empty keys (after stripping) are rejected."""
    with pytest.raises(ValueError, match="empty key"):
        key_val_to_dict("=1")

    with pytest.raises(ValueError, match="empty key"):
        key_val_to_dict("a=1, =2")


def test_key_val_to_dict_preserves_value_whitespace():
    """Test that value whitespace is preserved (not stripped), only keys are stripped."""
    result = key_val_to_dict("key = value with spaces")

    # Key is stripped, but value keeps leading space (from split on =)
    assert result == {"key": " value with spaces"}


def test_key_val_to_dict_handles_quoted_values_with_spaces():
    """Test that quoted values with spaces are handled correctly."""
    result = key_val_to_dict('a="hello world", b=2')

    assert result == {"a": "hello world", "b": 2}
