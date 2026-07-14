import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lm_eval.models.api_models import create_image_prompt
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


@pytest.fixture
def api_batch_ssl_tokenized():
    return LocalCompletionsAPI(
        base_url="https://test-url.com",
        model="EleutherAI/pythia-1b",
        verify_certificate=False,
        num_concurrent=2,
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


@pytest.mark.parametrize(
    "model_cls",
    [
        pytest.param(
            "LocalChatCompletion",
            id="local-chat-completions",
        ),
        pytest.param(
            "OpenAIChatCompletion",
            id="openai-chat-completions",
        ),
    ],
)
def test_chat_template_payload_does_not_add_top_level_type(model_cls):
    from lm_eval.models import openai_completions

    model = getattr(openai_completions, model_cls)(
        base_url="http://test-url.com",
        model="test-model",
    )
    chat = [{"role": "user", "content": "Reply with one word: hello"}]

    messages = model.create_message((model.apply_chat_template(chat),))
    payload = model._create_payload(messages, generate=True, gen_kwargs={})

    assert payload["messages"] == chat
    assert "type" not in payload["messages"][0]


@pytest.mark.parametrize("include_legacy_type", [False, True])
def test_create_image_prompt_uses_content_parts_without_top_level_type(
    include_legacy_type,
):
    class DummyImage:
        def save(self, buf, format):
            buf.write(b"image-bytes")

    chat = [{"role": "user", "content": "Describe this image"}]
    if include_legacy_type:
        chat[0]["type"] = "text"

    messages = create_image_prompt([DummyImage()], json.loads(json.dumps(chat)))

    assert "type" not in messages[-1]
    assert messages[-1]["content"][0]["type"] == "image_url"
    assert messages[-1]["content"][1] == {
        "type": "text",
        "text": "Describe this image",
    }


def test_parse_logplikelihood_greedy_and_non_greedy():
    outputs = {
        "choices": [
            {
                "index": 0,
                "logprobs": {
                    "token_logprobs": [None, -0.1, -0.2, -0.3],
                    "top_logprobs": [
                        {},
                        {"a": -0.1, "b": -5.0},
                        {"c": -0.2, "d": -5.0},
                        {"e": -1.0, "f": -0.5},
                    ],
                },
            }
        ]
    }
    # ctxlen=1 means the first token is context, slice is [1:-1] -> indices 1 and 2
    result = LocalCompletionsAPI.parse_logprobs(outputs, ctxlens=[1])

    assert len(result) == 1
    logprob, is_greedy = result[0]
    assert logprob == pytest.approx(-0.1 + -0.2)
    # both sampled tokens match the max of their top_logprobs, so greedy
    assert is_greedy is True


def test_parse_logplikelihood_detects_non_greedy():
    outputs = {
        "choices": [
            {
                "index": 0,
                "logprobs": {
                    "token_logprobs": [None, -2.0, -0.5],
                    "top_logprobs": [
                        {},
                        {"a": -0.1, "b": -2.0},
                        {"c": -0.5, "d": -5.0},
                    ],
                },
            }
        ]
    }
    result = LocalCompletionsAPI.parse_logprobs(outputs, ctxlens=[1])

    _, is_greedy = result[0]
    # first sampled token (-2.0) is not the max (-0.1), so not greedy
    assert is_greedy is False


def test_parse_logplikelihood_orders_choices_by_index():
    outputs = {
        "choices": [
            {
                "index": 1,
                "logprobs": {
                    "token_logprobs": [None, -0.5, -0.1],
                    "top_logprobs": [{}, {"a": -0.5}, {"b": -0.1}],
                },
            },
            {
                "index": 0,
                "logprobs": {
                    "token_logprobs": [None, -0.2, -0.3],
                    "top_logprobs": [{}, {"a": -0.2}, {"b": -0.3}],
                },
            },
        ]
    }
    result = LocalCompletionsAPI.parse_logprobs(outputs, ctxlens=[1, 1])

    # choices are sorted by index before being zipped with ctxlens
    assert result[0][0] == pytest.approx(-0.2)
    assert result[1][0] == pytest.approx(-0.5)


def test_parse_generations_orders_by_index():
    outputs = {
        "choices": [
            {"index": 1, "text": "world"},
            {"index": 0, "text": "hello"},
        ]
    }
    result = LocalCompletionsAPI.parse_generations(outputs)

    assert result == ["hello", "world"]


def test_chat_parse_generations_reads_message_content():
    from lm_eval.models.openai_completions import LocalChatCompletion

    outputs = {
        "choices": [
            {"index": 0, "message": {"content": "hello"}},
            {"index": 1, "message": {"content": "world"}},
        ]
    }
    result = LocalChatCompletion.parse_generations(outputs)

    assert result == ["hello", "world"]


def test_chat_parse_generations_handles_content_filter():
    from lm_eval.models.openai_completions import LocalChatCompletion

    # missing "content" simulates a response blocked by a content filter
    outputs = {"choices": [{"index": 0, "message": {}}]}
    result = LocalChatCompletion.parse_generations(outputs)

    assert result == [""]


@pytest.mark.parametrize(
    "model_name",
    ["o1-mini", "o3-mini", "o4-mini", "gpt-5"],
)
def test_openai_chat_payload_drops_stop_for_reasoning_models(model_name):
    from lm_eval.models.openai_completions import OpenAIChatCompletion

    model = OpenAIChatCompletion(base_url="http://test-url.com", model=model_name)
    messages = [{"role": "user", "content": "hi"}]
    payload = model._create_payload(messages, generate=True, gen_kwargs={})

    assert "stop" not in payload
    assert payload["temperature"] == 1
    assert payload["model"] == model_name


def test_openai_chat_payload_keeps_stop_for_standard_models():
    from lm_eval.models.openai_completions import OpenAIChatCompletion

    model = OpenAIChatCompletion(base_url="http://test-url.com", model="gpt-4")
    messages = [{"role": "user", "content": "hi"}]
    payload = model._create_payload(
        messages, generate=True, gen_kwargs={"until": ["END"], "temperature": 0.5}
    )

    assert payload["stop"] == ["END", "<|endoftext|>"]
    assert payload["temperature"] == 0.5
    assert payload["max_completion_tokens"] == model._max_gen_toks


class DummyAsyncContextManager:
    def __init__(self, result):
        self.result = result

    async def __aenter__(self):
        return self.result

    async def __aexit__(self, exc_type, exc, tb):
        pass


@pytest.mark.parametrize(
    "expected_inputs, expected_ctxlens, expected_cache_keys",
    [
        (
            [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
            ],
            [3, 3, 3, 3],
            ["cache_key1", "cache_key2", "cache_key3", "cache_key4"],
        ),
    ],
)
def test_get_batched_requests_with_no_ssl(
    api_batch_ssl_tokenized, expected_inputs, expected_ctxlens, expected_cache_keys
):
    with (
        patch(
            "lm_eval.models.api_models.TCPConnector", autospec=True
        ) as mock_connector,
        patch(
            "lm_eval.models.api_models.ClientSession", autospec=True
        ) as mock_client_session,
        patch(
            "lm_eval.models.openai_completions.LocalCompletionsAPI.parse_logprobs",
            autospec=True,
        ) as mock_parse,
    ):
        mock_session_instance = AsyncMock()
        mock_post_response = AsyncMock()
        mock_post_response.status = 200
        mock_post_response.ok = True
        mock_post_response.json = AsyncMock(return_value={"mocked": "response"})
        mock_post_response.raise_for_status = lambda: None
        mock_session_instance.post = lambda *args, **kwargs: DummyAsyncContextManager(
            mock_post_response
        )
        mock_client_session.return_value.__aenter__.return_value = mock_session_instance
        mock_parse.return_value = [(1.23, True), (4.56, False)]

        async def run():
            return await api_batch_ssl_tokenized.get_batched_requests(
                expected_inputs,
                expected_cache_keys,
                generate=False,
                ctxlens=expected_ctxlens,
            )

        result_batches = asyncio.run(run())

        mock_connector.assert_called_with(limit=2, ssl=False)
        assert result_batches


def test_local_completionsapi_remote_tokenizer_authenticated(monkeypatch):
    captured = {}

    class DummyTokenizer:
        def __init__(
            self, base_url, timeout, verify_certificate, ca_cert_path, auth_token
        ):
            captured.update(locals())

    monkeypatch.setattr("lm_eval.utils.RemoteTokenizer", DummyTokenizer)
    LocalCompletionsAPI(
        base_url="https://secure-server",
        tokenizer_backend="remote",
        verify_certificate=True,
        ca_cert_path="secure.crt",
        auth_token="secure-token",
    )
    assert captured["base_url"] == "https://secure-server"
    assert captured["verify_certificate"] is True
    assert captured["ca_cert_path"] == "secure.crt"
    assert captured["auth_token"] == "secure-token"


def test_local_completionsapi_remote_tokenizer_unauthenticated(monkeypatch):
    captured = {}

    class DummyTokenizer:
        def __init__(
            self, base_url, timeout, verify_certificate, ca_cert_path, auth_token
        ):
            captured.update(locals())

    monkeypatch.setattr("lm_eval.utils.RemoteTokenizer", DummyTokenizer)
    LocalCompletionsAPI(
        base_url="http://localhost:8000",
        tokenizer_backend="remote",
        verify_certificate=False,
        ca_cert_path=None,
        auth_token=None,
    )
    assert captured["base_url"] == "http://localhost:8000"
    assert captured["verify_certificate"] is False
    assert captured["ca_cert_path"] is None
    assert captured["auth_token"] is None


def test_localchatcompletion_remote_tokenizer_authenticated(monkeypatch):
    captured = {}

    class DummyTokenizer:
        def __init__(
            self, base_url, timeout, verify_certificate, ca_cert_path, auth_token
        ):
            captured.update(locals())

    monkeypatch.setattr("lm_eval.utils.RemoteTokenizer", DummyTokenizer)
    from lm_eval.models.openai_completions import LocalChatCompletion

    LocalChatCompletion(
        base_url="https://secure-server",
        tokenizer_backend="remote",
        verify_certificate=True,
        ca_cert_path="secure.crt",
        auth_token="secure-token",
    )
    assert captured["base_url"] == "https://secure-server"
    assert captured["verify_certificate"] is True
    assert captured["ca_cert_path"] == "secure.crt"
    assert captured["auth_token"] == "secure-token"


def test_localchatcompletion_remote_tokenizer_unauthenticated(monkeypatch):
    captured = {}

    class DummyTokenizer:
        def __init__(
            self, base_url, timeout, verify_certificate, ca_cert_path, auth_token
        ):
            captured.update(locals())

    monkeypatch.setattr("lm_eval.utils.RemoteTokenizer", DummyTokenizer)
    from lm_eval.models.openai_completions import LocalChatCompletion

    LocalChatCompletion(
        base_url="http://localhost:8000",
        tokenizer_backend="remote",
        verify_certificate=False,
        ca_cert_path=None,
        auth_token=None,
    )
    assert captured["base_url"] == "http://localhost:8000"
    assert captured["verify_certificate"] is False
    assert captured["ca_cert_path"] is None
    assert captured["auth_token"] is None
