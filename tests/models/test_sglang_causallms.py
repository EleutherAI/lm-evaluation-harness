from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from lm_eval.api.instance import Instance
from lm_eval.models.sglang_causallms import SGLangLM


@pytest.mark.parametrize(
    ("batch_size", "expected_batches"),
    [
        (1, [[[3, 4, 5, 6]], [[9, 8]]]),
        (2, [[[3, 4, 5, 6], [9, 8]]]),
    ],
)
def test_sglang_scores_continuations_and_restores_input_order(
    batch_size, expected_batches
):
    long_response = {
        "meta_info": {
            "input_token_logprobs": [
                (None, 3, None),
                (-9.0, 4, None),
                (-0.25, 5, None),
                (-1.0, 6, None),
            ],
            "input_top_logprobs": [
                None,
                [(-0.125, 0, None)],
                [(-1.0, 0, None), (-0.25, 5, None)],
                [(-2.0, 0, None), (-1.0, 6, None)],
            ],
        }
    }
    short_response = {
        "meta_info": {
            "input_token_logprobs": [(None, 9, None), (-0.75, 8, None)],
            "input_top_logprobs": [None, [(-0.75, 8, None), (-0.25, 7, None)]],
        }
    }
    responses_by_input = {
        (3, 4, 5, 6): long_response,
        (9, 8): short_response,
    }

    def generate(input_ids, **kwargs):
        return [responses_by_input[tuple(tokens)] for tokens in input_ids]

    model = SGLangLM.__new__(SGLangLM)
    model.batch_size = batch_size
    model._max_length = 4
    model.cache_hook = Mock()
    model.model = SimpleNamespace(generate=Mock(side_effect=generate))

    cache_key = ("context", " continuation")
    requests = [
        (None, [9], [8]),
        (cache_key, [1, 2, 3, 4], [5, 6]),
    ]

    assert model._loglikelihood_tokens(requests, disable_tqdm=True) == [
        (-0.75, False),
        (-1.25, True),
    ]
    assert model.model.generate.call_args_list == [
        call(
            input_ids=batch,
            sampling_params=[{"temperature": 0, "max_new_tokens": 1}] * len(batch),
            return_logprob=True,
            top_logprobs_num=2,
            logprob_start_len=0,
        )
        for batch in expected_batches
    ]
    model.cache_hook.add_partial.assert_called_once_with(
        "loglikelihood", cache_key, (-1.25, True)
    )


def test_sglang_rolling_scores_windows_per_document_and_caches_totals():
    tokenized = {"short": [8], "long": [1, 2, 3, 4, 5], "": []}

    def generate(input_ids, **kwargs):
        return [
            {
                "meta_info": {
                    "input_token_logprobs": [(None, tokens[0], None)]
                    + [(-float(token), token, None) for token in tokens[1:]],
                    "input_top_logprobs": [None]
                    + [[(-float(token), token, None)] for token in tokens[1:]],
                }
            }
            for tokens in input_ids
        ]

    model = SGLangLM.__new__(SGLangLM)
    model.batch_size = 2
    model._max_length = 4
    model.custom_prefix_token_id = 99
    model.add_bos_token = False
    model.tokenizer = Mock(
        side_effect=lambda text, **kwargs: SimpleNamespace(input_ids=tokenized[text])
    )
    model.cache_hook = Mock()
    model.model = SimpleNamespace(generate=Mock(side_effect=generate))
    requests = [
        Instance("loglikelihood_rolling", {}, (text,), idx)
        for idx, text in enumerate(tokenized)
    ]

    assert model.loglikelihood_rolling(requests, disable_tqdm=True) == [
        -8.0,
        -15.0,
        0.0,
    ]
    assert [
        args.kwargs["input_ids"] for args in model.model.generate.call_args_list
    ] == [
        [[99, 1, 2, 3], [99, 8]],
        [[2, 3, 4, 5]],
    ]
    assert model.cache_hook.add_partial.call_args_list == [
        call("loglikelihood_rolling", ("short",), -8.0),
        call("loglikelihood_rolling", ("long",), -15.0),
        call("loglikelihood_rolling", ("",), 0.0),
    ]
