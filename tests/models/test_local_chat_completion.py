import pytest

from lm_eval.models.openai_completions import LocalChatCompletion


@pytest.mark.parametrize(
    "think_end_token, content, expected",
    [
        (
            None,
            "<think>reasoning</think> final answer",
            "<think>reasoning</think> final answer",
        ),
        ("</think>", "<think>reasoning</think>  final answer", "final answer"),
        ("</think>", "first</think> draft</think> final answer", "final answer"),
        (
            "</think>",
            "answer without a thinking marker",
            "answer without a thinking marker",
        ),
        ("</think>", None, None),
    ],
)
def test_parse_generations_strips_thinking(think_end_token, content, expected):
    model = LocalChatCompletion(
        base_url="http://test-url.com",
        model="test-model",
        think_end_token=think_end_token,
    )
    response = {
        "choices": [{"index": 0, "message": {"content": content}}],
    }

    assert model.parse_generations(response) == [expected]
