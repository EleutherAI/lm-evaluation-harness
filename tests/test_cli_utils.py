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
