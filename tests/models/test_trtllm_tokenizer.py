"""Offline tokenizer contracts for the real TensorRT-LLM adapter."""

from unittest.mock import Mock

import pytest
from jinja2.exceptions import TemplateError
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from lm_eval.models import trtllm_causallms


UNK = "[UNK]"
BOS = "[BOS]"
EOS = "[EOS]"


@pytest.fixture
def tokenizer():
    backend = Tokenizer(
        models.WordLevel(
            {UNK: 0, BOS: 1, EOS: 2, "hello": 3, "world": 4, "test": 5},
            unk_token=UNK,
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="[BOS] $A", special_tokens=[(BOS, 1)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token=UNK,
        bos_token=BOS,
        eos_token=EOS,
    )


@pytest.fixture
def make_adapter(monkeypatch, tokenizer):
    monkeypatch.setattr(trtllm_causallms, "find_spec", lambda _: object())
    monkeypatch.setattr(trtllm_causallms, "LLM", Mock(), raising=False)
    monkeypatch.setattr(trtllm_causallms, "KvCacheConfig", Mock(), raising=False)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", Mock(return_value=tokenizer))

    def make(**kwargs):
        return trtllm_causallms.TRTLLM(model="offline-model", **kwargs)

    return make


@pytest.mark.parametrize("tokenizer_adds_bos", [True, False])
def test_detects_tokenizer_bos_behavior(make_adapter, tokenizer, tokenizer_adds_bos):
    if not tokenizer_adds_bos:
        tokenizer.backend_tokenizer.post_processor = processors.TemplateProcessing(
            single="$A"
        )

    adapter = make_adapter()

    assert adapter.add_bos_token is tokenizer_adds_bos
    assert adapter.tok_encode("hello world") == (
        [1, 3, 4] if tokenizer_adds_bos else [3, 4]
    )
    assert tokenizer.pad_token_id == tokenizer.unk_token_id


@pytest.mark.parametrize("add_bos", [True, False])
def test_explicit_bos_option_overrides_detection(make_adapter, add_bos):
    adapter = make_adapter(add_bos_token=add_bos)

    assert adapter.add_bos_token is add_bos
    assert adapter.tok_encode("hello") == ([1, 3] if add_bos else [3])


def test_mixed_bos_batch_preserves_order_without_duplicate_prefix(make_adapter):
    adapter = make_adapter()

    encoded = adapter.tok_encode(["world", "[BOS] hello", "hello world", "[BOS] world"])

    assert encoded == [[1, 4], [1, 3], [1, 3, 4], [1, 4]]
    assert adapter.tok_encode("[BOS] hello") == [1, 3]


@pytest.mark.parametrize("add_bos", [True, False])
@pytest.mark.parametrize("add_special_tokens", [True, False])
def test_per_call_special_token_option_wins(make_adapter, add_bos, add_special_tokens):
    adapter = make_adapter(add_bos_token=add_bos)

    assert adapter.tok_encode("hello", add_special_tokens=add_special_tokens) == (
        [1, 3] if add_special_tokens else [3]
    )
    assert adapter.tok_encode("[BOS] hello", add_special_tokens=add_special_tokens) == [
        1,
        3,
    ]


@pytest.mark.parametrize("text", ["", []])
def test_empty_input_produces_no_tokens(make_adapter, text):
    assert make_adapter().tok_encode(text) == []


@pytest.fixture
def chat_tokenizer(tokenizer):
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ message['role'] }}:{{ message['content'] }}|"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "assistant:{{ style }}:{{ enable_thinking }}"
        "{% endif %}"
    )
    tokenizer.apply_chat_template = Mock(wraps=tokenizer.apply_chat_template)
    return tokenizer


@pytest.mark.parametrize("add_generation_prompt", [True, False])
@pytest.mark.parametrize("enable_thinking", [True, False])
def test_chat_template_forwards_options_and_continuation_mode(
    make_adapter, chat_tokenizer, add_generation_prompt, enable_thinking
):
    adapter = make_adapter(
        enable_thinking=enable_thinking, chat_template_args={"style": "brief"}
    )
    history = [{"role": "user", "content": "hello"}]

    rendered = adapter.apply_chat_template(history, add_generation_prompt)

    expected = (
        f"user:hello|assistant:brief:{enable_thinking}"
        if add_generation_prompt
        else "user:hello"
    )
    assert rendered == expected
    chat_tokenizer.apply_chat_template.assert_called_once_with(
        history,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=not add_generation_prompt,
        enable_thinking=enable_thinking,
        style="brief",
    )


def test_chat_template_args_override_thinking_default(make_adapter, chat_tokenizer):
    adapter = make_adapter(
        enable_thinking=True,
        chat_template_args={"enable_thinking": False, "style": "brief"},
    )

    rendered = adapter.apply_chat_template([{"role": "user", "content": "hello"}])

    assert rendered == "user:hello|assistant:brief:False"
    assert (
        chat_tokenizer.apply_chat_template.call_args.kwargs["enable_thinking"] is False
    )


def test_chat_template_retries_without_system_messages(
    make_adapter, chat_tokenizer, caplog
):
    chat_tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ raise_exception('system role is unsupported') }}"
        "{% endif %}"
        "{{ message['role'] }}:{{ message['content'] }}|"
        "{% endfor %}"
        "{% if add_generation_prompt %}assistant:{{ style }}:{{ enable_thinking }}{% endif %}"
    )
    adapter = make_adapter(enable_thinking=False, chat_template_args={"style": "brief"})
    history = [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
        {"role": "system", "content": "more instructions"},
        {"role": "user", "content": "test"},
    ]
    original_history = [message.copy() for message in history]

    rendered = adapter.apply_chat_template(history)

    assert rendered == "user:hello|assistant:world|user:test|assistant:brief:False"
    assert history == original_history
    first_call, retry = chat_tokenizer.apply_chat_template.call_args_list
    assert first_call.args == (history,)
    assert retry.args == (
        [message for message in history if message["role"] != "system"],
    )
    assert retry.kwargs == first_call.kwargs
    assert "Removing the system role" in caplog.text


def test_bos_probe_failure_disables_automatic_special_tokens(
    make_adapter, tokenizer, monkeypatch
):
    with monkeypatch.context() as probe:
        probe.setattr(
            tokenizer, "decode", Mock(side_effect=ValueError("cannot decode"))
        )
        adapter = make_adapter()

    assert adapter.add_bos_token is False
    assert adapter.tok_encode("hello") == [3]


def test_chat_template_does_not_retry_unrelated_errors(make_adapter, chat_tokenizer):
    adapter = make_adapter()
    error = ValueError("invalid tokenizer configuration")
    chat_tokenizer.apply_chat_template.side_effect = error

    with pytest.raises(ValueError, match="invalid tokenizer configuration") as caught:
        adapter.apply_chat_template([{"role": "user", "content": "hello"}])

    assert caught.value is error
    assert chat_tokenizer.apply_chat_template.call_count == 1


def test_chat_template_propagates_failure_of_the_system_free_retry(
    make_adapter, chat_tokenizer
):
    adapter = make_adapter()
    retry_error = TemplateError("user role is also unsupported")
    chat_tokenizer.apply_chat_template.side_effect = [
        TemplateError("system role is unsupported"),
        retry_error,
    ]

    with pytest.raises(TemplateError, match="user role is also unsupported") as caught:
        adapter.apply_chat_template(
            [
                {"role": "system", "content": "instructions"},
                {"role": "user", "content": "hello"},
            ]
        )

    assert caught.value is retry_error
    assert chat_tokenizer.apply_chat_template.call_count == 2
