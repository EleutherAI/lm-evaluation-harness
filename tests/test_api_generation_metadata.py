"""Tests for the generation metadata logged by the OpenAI-compatible clients."""

TRACE_END = "[/THINK]"

from lm_eval.models.openai_completions import (
    CompletionText,
    LocalChatCompletion,
    LocalCompletionsAPI,
)


def make_lm(trace_end=None, eos_string=None):
    """A client without an endpoint: only the metadata helpers are exercised."""
    lm = object.__new__(LocalCompletionsAPI)
    lm.think_end_token = trace_end
    lm.eos_string = eos_string
    lm.tokenizer = None
    lm._max_gen_toks = 256
    lm.model = "fake"
    return lm


def completion_choice(**kwargs):
    choice = {"finish_reason": "stop", "stop_reason": None}
    choice.update(kwargs)
    return choice


def test_stop_reason_for_a_truncated_generation():
    lm = make_lm()
    result = lm._to_completion_text(
        "17 x 23 =", completion_choice(finish_reason="length")
    )
    assert result.stop_reason == "max_gen_toks"


def test_stop_reason_for_a_matched_stop_sequence():
    lm = make_lm()
    result = lm._to_completion_text(
        "391[INST]", completion_choice(stop_reason="[INST]"), stop=["[INST]"]
    )
    assert result.stop_reason == "stop_sequence:[INST]"


def test_stop_reason_for_a_natural_eos():
    lm = make_lm(eos_string="</s>")
    result = lm._to_completion_text("391", completion_choice())
    assert result.stop_reason == "stop_sequence:</s>"

    # without a known eos the plain vocabulary spelling is used
    assert (
        make_lm()._to_completion_text("391", completion_choice()).stop_reason
        == "natural"
    )


def test_full_resp_keeps_the_untrimmed_generation():
    lm = make_lm(trace_end=TRACE_END)
    raw = "[THINK]let me think[/THINK]391"
    result = lm._to_completion_text(raw, completion_choice(), stop=None)

    assert isinstance(result, CompletionText)
    assert str(result) == "391"  # what the task scores
    assert result.full_resp == raw  # what generation-length measures
    assert result.answer_found is True


def test_answer_found_is_false_when_the_trace_never_closed():
    lm = make_lm(trace_end=TRACE_END)
    raw = "[THINK]still thinking about it"
    result = lm._to_completion_text(
        raw, completion_choice(finish_reason="length"), stop=None
    )

    assert result.full_resp == raw
    assert result.answer_found is False


def test_think_end_token_drops_the_task_stops_from_the_request():
    lm = make_lm(trace_end=TRACE_END)
    payload = lm._create_payload(
        "prompt", generate=True, gen_kwargs={"until": ["[INST]", "</s>"]}, eos="</s>"
    )

    # task stops can match inside the reasoning trace, so only the eos is sent
    assert payload["stop"] == ["</s>"]

    plain = make_lm()._create_payload(
        "prompt", generate=True, gen_kwargs={"until": ["[INST]"]}, eos="</s>"
    )
    assert "[INST]" in plain["stop"]


def test_chat_client_logs_the_same_metadata():
    lm = object.__new__(LocalChatCompletion)
    lm.think_end_token = TRACE_END
    lm.eos_string = None
    lm.tokenizer = None

    outputs = {
        "choices": [
            {
                "index": 0,
                "finish_reason": "length",
                "stop_reason": None,
                "message": {"content": "[THINK]working on it"},
            }
        ]
    }
    (result,) = lm.parse_generations(outputs=outputs)

    assert isinstance(result, CompletionText)
    assert result.stop_reason == "max_gen_toks"
    assert result.full_resp == "[THINK]working on it"
    assert result.answer_found is False
