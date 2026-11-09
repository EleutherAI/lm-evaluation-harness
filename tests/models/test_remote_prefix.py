from types import SimpleNamespace
from unittest.mock import patch

import pytest

from lm_eval.api.instance import Instance
from lm_eval.models.openai_completions import LocalCompletionsAPI


@pytest.mark.parametrize(
    "bos_id,eos_id,custom_id,tokens,expected_context,expected_continuation",
    [
        (0, 1, None, [10, 11], [0], [10, 11]),
        (0, None, None, [10, 11], [0], [10, 11]),
        (5, 1, None, [10, 11], [5], [10, 11]),
        (None, 1, None, [10, 11], [1], [10, 11]),
        (None, 0, None, [10, 11], [0], [10, 11]),
        (5, 1, 0, [10, 11], [0], [10, 11]),
        (0, 1, 7, [10, 11], [7], [10, 11]),
        (0, 1, None, [0, 10], [0], [10]),
    ],
    ids=[
        "zero-bos",
        "zero-bos-without-eos",
        "nonzero-bos",
        "missing-bos",
        "zero-eos-fallback",
        "explicit-zero-prefix",
        "explicit-nonzero-prefix",
        "continuation-already-starts-with-bos",
    ],
)
def test_remote_prefix_conditions_empty_context(
    bos_id, eos_id, custom_id, tokens, expected_context, expected_continuation
):
    api = LocalCompletionsAPI(
        model="fixture",
        tokenizer_backend=None,
        custom_prefix_token_id=custom_id,
        header={"X-LM-Eval-Test": "offline"},
    )
    api.tokenizer_backend = "remote"
    api.tokenizer = SimpleNamespace(bos_token_id=bos_id, eos_token_id=eos_id)
    request = Instance(
        request_type="loglikelihood", doc={}, arguments=("", "answer"), idx=0
    )

    with (
        patch.object(api, "tok_encode", return_value=tokens),
        patch.object(
            api, "_loglikelihood_tokens", return_value=[(-1.0, True)]
        ) as score,
    ):
        assert api.loglikelihood([request], disable_tqdm=True) == [(-1.0, True)]

    score.assert_called_once_with(
        [(("", "answer"), expected_context, expected_continuation)],
        disable_tqdm=True,
    )


def test_remote_zero_bos_conditions_first_rolling_window():
    api = LocalCompletionsAPI(
        model="fixture",
        tokenizer_backend=None,
        max_length=4,
        header={"X-LM-Eval-Test": "offline"},
    )
    api.tokenizer_backend = "remote"
    api.tokenizer = SimpleNamespace(bos_token_id=0, eos_token_id=1)
    request = Instance(
        request_type="loglikelihood_rolling", doc={}, arguments=("text",), idx=0
    )

    with (
        patch.object(api, "tok_encode", return_value=[10, 11, 12]),
        patch.object(
            api, "_loglikelihood_tokens", return_value=[(-0.2, True), (-0.3, True)]
        ) as score,
    ):
        assert api.loglikelihood_rolling([request], disable_tqdm=True) == [-0.5]

    score.assert_called_once_with(
        [(None, [0], [10, 11]), (None, [10, 11], [12])], disable_tqdm=True
    )
