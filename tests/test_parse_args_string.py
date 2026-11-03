import pytest

from lm_eval.utils import simple_parse_args_string


@pytest.mark.parametrize("args_string", ["a=1,b", "a=1,,b=2", "=2", "pretrained=gpt2,trust_remote_code"])
def test_simple_parse_args_string_rejects_malformed_entries(args_string):
    with pytest.raises(ValueError, match="Malformed argument string"):
        simple_parse_args_string(args_string)


def test_simple_parse_args_string_keeps_working_for_valid_entries():
    assert simple_parse_args_string("pretrained=gpt2,batch_size=8") == {
        "pretrained": "gpt2",
        "batch_size": 8,
    }


def test_simple_parse_args_string_handles_empty():
    assert simple_parse_args_string(None) == {}
    assert simple_parse_args_string("   ") == {}
