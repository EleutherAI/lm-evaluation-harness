import pytest

from lm_eval.api.instance import Instance
from lm_eval.api.registry import get_model
from lm_eval.models.dummy import DummyLM


def _instance(request_type, arguments, idx=0):
    return Instance(request_type=request_type, doc={}, arguments=arguments, idx=idx)


def test_registered_under_dummy_name():
    assert get_model("dummy") is DummyLM


def test_create_from_arg_string_ignores_its_arguments():
    lm = DummyLM.create_from_arg_string("anything=goes", {"also": "ignored"})

    assert isinstance(lm, DummyLM)
    assert lm.write_out is False


def test_loglikelihood_returns_one_result_per_request():
    lm = DummyLM()
    requests = [
        _instance("loglikelihood", ("context one", "continuation one"), idx=0),
        _instance("loglikelihood", ("context two", "continuation two"), idx=1),
    ]

    results = lm.loglikelihood(requests, disable_tqdm=True)

    assert len(results) == len(requests)
    for logprob, is_greedy in results:
        assert -1.0 <= logprob <= 0.0
        assert is_greedy is False


def test_loglikelihood_rolling_returns_one_float_per_request():
    lm = DummyLM()
    requests = [
        _instance("loglikelihood_rolling", ("a passage of text",), idx=0),
        _instance("loglikelihood_rolling", ("another passage",), idx=1),
        _instance("loglikelihood_rolling", ("a third passage",), idx=2),
    ]

    results = lm.loglikelihood_rolling(requests, disable_tqdm=True)

    assert len(results) == len(requests)
    for logprob in results:
        assert -1.0 <= logprob <= 0.0


def test_generate_until_returns_lol_for_each_request():
    lm = DummyLM()
    requests = [
        _instance("generate_until", ("hello there", {"until": ["\n"]}), idx=0),
        _instance("generate_until", ("another prompt", {}), idx=1),
    ]

    results = lm.generate_until(requests, disable_tqdm=True)

    assert results == ["lol", "lol"]


def test_generate_until_rejects_blank_context():
    lm = DummyLM()
    requests = [_instance("generate_until", ("   ", {}), idx=0)]

    with pytest.raises(AssertionError):
        lm.generate_until(requests, disable_tqdm=True)


def test_write_out_prints_context_and_continuation(capsys):
    lm = DummyLM(write_out=True)
    requests = [_instance("loglikelihood", ("my context", "my continuation"), idx=0)]

    lm.loglikelihood(requests, disable_tqdm=True)

    captured = capsys.readouterr()
    assert "context: my context" in captured.out
    assert "continuation: my continuation" in captured.out


def test_write_out_prints_prompt_and_gen_kwargs_for_generate_until(capsys):
    lm = DummyLM(write_out=True)
    requests = [_instance("generate_until", ("my prompt", {"until": ["\n"]}), idx=0)]

    lm.generate_until(requests, disable_tqdm=True)

    captured = capsys.readouterr()
    assert "my prompt" in captured.out
    assert "gen_kwargs: {'until': ['\\n']}" in captured.out


class _FakeTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(
        self, chat_history, tokenize, add_generation_prompt, continue_final_message
    ):
        self.calls.append(
            {
                "chat_history": chat_history,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "continue_final_message": continue_final_message,
            }
        )
        return "rendered-prompt"


def test_apply_chat_template_defaults_to_generation_prompt():
    lm = DummyLM()
    fake_tokenizer = _FakeTokenizer()
    lm.tokenizer = fake_tokenizer
    chat_history = [{"role": "user", "content": "hi"}]

    rendered = lm.apply_chat_template(chat_history)

    assert rendered == "rendered-prompt"
    assert fake_tokenizer.calls == [
        {
            "chat_history": chat_history,
            "tokenize": False,
            "add_generation_prompt": True,
            "continue_final_message": False,
        }
    ]


def test_apply_chat_template_continues_final_message_when_no_generation_prompt():
    lm = DummyLM()
    fake_tokenizer = _FakeTokenizer()
    lm.tokenizer = fake_tokenizer
    chat_history = [{"role": "assistant", "content": "partial answer"}]

    lm.apply_chat_template(chat_history, add_generation_prompt=False)

    assert fake_tokenizer.calls[0]["add_generation_prompt"] is False
    assert fake_tokenizer.calls[0]["continue_final_message"] is True
