import pytest

from lm_eval._cli.utils import (
    handle_cli_value_string,
    key_val_to_dict,
    split_top_level,
)


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


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        # An apostrophe inside a word is a literal character, not an opening
        # quote: it must not swallow the separators that follow it.
        ("until=Bob's,temperature=0", ["until=Bob's", "temperature=0"]),
        ("a=don't,b=won't,c=1", ["a=don't", "b=won't", "c=1"]),
        ("a=5'", ["a=5'"]),
        # An unterminated quote degrades to a literal character rather than
        # consuming the rest of the string.
        ('a="unterminated,b=1', ['a="unterminated', "b=1"]),
        # A quote that starts a token still protects the commas inside it.
        ("a='x,y',b=2", ["a='x,y'", "b=2"]),
        ('a="x,y",b=2', ['a="x,y"', "b=2"]),
        ('desc=say "hi",temperature=0', ['desc=say "hi"', "temperature=0"]),
        # Both forms combined: literal apostrophe, then a genuinely quoted value.
        ("a=don't stop,b='x,y'", ["a=don't stop", "b='x,y'"]),
        # Brackets and braces keep protecting their own commas.
        (
            "max_seq_lengths=[4096,8192],tokenizer=gpt2",
            ["max_seq_lengths=[4096,8192]", "tokenizer=gpt2"],
        ),
        (
            "chat_template_args={'reasoning_effort':'low'},dtype=auto",
            ["chat_template_args={'reasoning_effort':'low'}", "dtype=auto"],
        ),
    ],
)
def test_split_top_level_treats_in_word_quotes_as_literal(args, expected):
    assert split_top_level(args) == expected


def test_key_val_to_dict_keeps_pairs_after_an_apostrophe():
    result = key_val_to_dict("until=Bob's,temperature=0,max_gen_toks=256")

    assert result == {"until": "Bob's", "temperature": 0, "max_gen_toks": 256}
