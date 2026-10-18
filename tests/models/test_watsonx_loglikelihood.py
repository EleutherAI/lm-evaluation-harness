import pytest

from lm_eval.models.ibm_watsonx_ai import LogLikelihoodResult, WatsonxLLM


@pytest.fixture
def watsonx():
    # Scoring an existing response requires no SDK client or model connection.
    return WatsonxLLM.__new__(WatsonxLLM)


@pytest.mark.parametrize(
    "context, response, expected",
    [
        (
            ["prompt"],
            [("prompt", -9.0, 2), (" answer", -0.25, 1)],
            (-0.25, True),
        ),
        (
            ["first", " prompt"],
            [("first", -8.0, 2), (" prompt", -9.0, 3), (" answer", -0.25, 1)],
            (-0.25, True),
        ),
        (
            ["prompt"],
            [("prompt", -9.0, 2), (" answer", -0.25, 1), ("prompt", -0.5, 1)],
            (-0.75, True),
        ),
        (
            ["prompt"],
            [("prompt", -9.0, 1), (" answer", -0.25, 2)],
            (-0.25, False),
        ),
        (
            ["prompt", "</s>"],
            [("prompt", -9.0, 2), (" answer", -0.25, 1)],
            (-0.25, True),
        ),
        (
            ["prompt", "</s>"],
            [("prompt", -9.0, 2), (" answer", -0.25, 1), ("</s>", -0.5, 1)],
            (-0.75, True),
        ),
        (["prompt"], [("prompt", -9.0, 2)], (0.0, True)),
        (["prompt", "</s>"], [("prompt", -9.0, 2)], (0.0, True)),
        ([], [("answer", -0.25, 1), (" text", -0.5, 2)], (-0.75, False)),
    ],
    ids=[
        "one-prompt-token",
        "multiple-prompt-tokens",
        "continuation-ends-with-prompt-token",
        "non-greedy-continuation",
        "tokenizer-stop-token",
        "continuation-ends-with-stop-token",
        "empty-continuation",
        "empty-continuation-after-tokenizer-stop",
        "empty-context",
    ],
)
def test_loglikelihood_scores_only_continuation(watsonx, context, response, expected):
    input_tokens = [
        {"text": text, "logprob": logprob, "rank": rank}
        for text, logprob, rank in response
    ]

    result = watsonx._get_log_likelihood(input_tokens, context)

    assert result == LogLikelihoodResult(*expected)


def test_loglikelihood_rejects_mismatched_context(watsonx):
    input_tokens = [
        {"text": "different", "logprob": -9.0, "rank": 1},
        {"text": " answer", "logprob": -0.25, "rank": 1},
    ]

    with pytest.raises(RuntimeError, match="unexpected difference"):
        watsonx._get_log_likelihood(input_tokens, ["prompt", " context"])
