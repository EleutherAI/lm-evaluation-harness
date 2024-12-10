from unittest.mock import MagicMock, patch

import pytest

from lm_eval.models.openai_completions import LocalCompletionsAPI


@pytest.fixture
def api():
    return LocalCompletionsAPI(
        base_url="http://test-url.com", tokenizer_backend=None, model="gpt-3.5-turbo"
    )


@pytest.fixture
def api_tokenized():
    return LocalCompletionsAPI(
        base_url="http://test-url.com",
        model="EleutherAI/pythia-1b",
        tokenizer_backend="huggingface",
    )


def test_create_payload_generate(api):
    messages = ["Generate a story"]
    gen_kwargs = {
        "max_tokens": 100,
        "temperature": 0.7,
        "until": ["The End"],
        "do_sample": True,
        "seed": 1234,
    }
    payload = api._create_payload(messages, generate=True, gen_kwargs=gen_kwargs)

    assert payload == {
        "prompt": ["Generate a story"],
        "model": "gpt-3.5-turbo",
        "max_tokens": 100,
        "temperature": 0.7,
        "stop": ["The End"],
        "seed": 1234,
    }


def test_create_payload_loglikelihood(api):
    messages = ["The capital of France is"]
    payload = api._create_payload(messages, generate=False, gen_kwargs=None)

    assert payload == {
        "model": "gpt-3.5-turbo",
        "prompt": ["The capital of France is"],
        "max_tokens": 1,
        "logprobs": 1,
        "echo": True,
        "temperature": 0,
        "seed": 1234,
    }


@pytest.mark.parametrize(
    "input_messages, generate, gen_kwargs, expected_payload",
    [
        (
            ["Hello, how are"],
            True,
            {"max_gen_toks": 100, "temperature": 0.7, "until": ["hi"]},
            {
                "prompt": "Hello, how are",
                "model": "gpt-3.5-turbo",
                "max_tokens": 100,
                "temperature": 0.7,
                "stop": ["hi"],
                "seed": 1234,
            },
        ),
        (
            ["Hello, how are", "you"],
            True,
            {},
            {
                "prompt": "Hello, how are",
                "model": "gpt-3.5-turbo",
                "max_tokens": 256,
                "temperature": 0,
                "stop": [],
                "seed": 1234,
            },
        ),
    ],
)
def test_model_generate_call_usage(
    api, input_messages, generate, gen_kwargs, expected_payload
):
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "success"}
        mock_post.return_value = mock_response

        # Act
        result = api.model_call(
            input_messages, generate=generate, gen_kwargs=gen_kwargs
        )

        # Assert
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert "json" in kwargs
        assert kwargs["json"] == expected_payload
        assert result == {"result": "success"}


@pytest.mark.parametrize(
    "input_messages, generate, gen_kwargs, expected_payload",
    [
        (
            [[1, 2, 3, 4, 5]],
            False,
            None,
            {
                "model": "EleutherAI/pythia-1b",
                "prompt": [[1, 2, 3, 4, 5]],
                "max_tokens": 1,
                "logprobs": 1,
                "echo": True,
                "seed": 1234,
                "temperature": 0,
            },
        ),
    ],
)
def test_model_tokenized_call_usage(
    api_tokenized, input_messages, generate, gen_kwargs, expected_payload
):
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "success"}
        mock_post.return_value = mock_response

        # Act
        result = api_tokenized.model_call(
            input_messages, generate=generate, gen_kwargs=gen_kwargs
        )

        # Assert
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert "json" in kwargs
        assert kwargs["json"] == expected_payload
        assert result == {"result": "success"}
