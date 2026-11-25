from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lm_eval.api.instance import Instance
from lm_eval.models.sglang_causallms import SGLangLM


def _make_request(continuation_length):
    context_tokens = [8, 9]
    continuation_tokens = list(range(1, continuation_length + 1))

    def generate(*, input_ids, **kwargs):
        return [
            {
                "meta_info": {
                    "input_token_logprobs": [(None, tokens[0], None)]
                    + [(-0.25, token, None) for token in tokens[1:]],
                    "input_top_logprobs": [None]
                    + [[(-0.25, token, None)] for token in tokens[1:]],
                }
            }
            for tokens in input_ids
        ]

    model = SGLangLM.__new__(SGLangLM)
    model._max_length = 4
    model.batch_size = 1
    model.cache_hook = Mock()
    model.model = SimpleNamespace(generate=Mock(side_effect=generate))
    model.tok_encode = Mock(
        side_effect={
            "context": context_tokens,
            "context continuation": context_tokens + continuation_tokens,
        }.__getitem__,
    )
    request = Instance("loglikelihood", {}, ("context", " continuation"), 0)
    return model, request


@pytest.mark.parametrize("continuation_length", [4, 5])
def test_loglikelihood_rejects_continuation_without_room_for_context(
    continuation_length,
):
    model, request = _make_request(continuation_length)

    with pytest.raises(ValueError, match="requires at least one context token"):
        model.loglikelihood([request], disable_tqdm=True)

    model.model.generate.assert_not_called()


def test_loglikelihood_retains_one_context_token_at_window_limit():
    model, request = _make_request(continuation_length=3)

    assert model.loglikelihood([request], disable_tqdm=True) == [(-0.75, True)]
    assert model.model.generate.call_args.kwargs["input_ids"] == [[9, 1, 2, 3]]
