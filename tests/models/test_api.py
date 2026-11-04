import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tenacity import wait_none

from lm_eval.api.instance import Instance
from lm_eval.models.api_models import create_image_prompt
from lm_eval.models.openai_completions import LocalCompletionsAPI
from lm_eval.models.sglang_generate_API import SGLANGGENERATEAPI


TEST_AUTH_TOKEN = "secure-token"  # noqa: S105 - fixed test sentinel


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


def test_model_call_passes_configured_timeout():
    api = LocalCompletionsAPI(
        base_url="http://test-url.com",
        tokenizer_backend=None,
        model="gpt-3.5-turbo",
        timeout=17,
    )

    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "success"}
        mock_post.return_value = mock_response

        api.model_call(["Hello"], generate=True, gen_kwargs={})

        assert mock_post.call_args.kwargs["timeout"] == 17


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
        auth_token=TEST_AUTH_TOKEN,
    )
    assert captured["base_url"] == "https://secure-server"
    assert captured["verify_certificate"] is True
    assert captured["ca_cert_path"] == "secure.crt"
    assert captured["auth_token"] == TEST_AUTH_TOKEN


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
        auth_token=TEST_AUTH_TOKEN,
    )
    assert captured["base_url"] == "https://secure-server"
    assert captured["verify_certificate"] is True
    assert captured["ca_cert_path"] == "secure.crt"
    assert captured["auth_token"] == TEST_AUTH_TOKEN


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


def _completions_model(num_concurrent=1, batch_size=1):
    return LocalCompletionsAPI(
        base_url="http://test-url.com",
        tokenizer_backend=None,
        model="gpt-3.5-turbo",
        num_concurrent=num_concurrent,
        batch_size=batch_size,
    )


def _repeated(prompt, gen_kwargs, repeats):
    # The evaluator repeats a request by passing the same Instance `repeats` times.
    instance = Instance("generate_until", {}, (prompt, gen_kwargs), 0, ("t", 0, 1))
    return [instance] * repeats


def _generate_with_fake_server(model, instances, fail_first_request=False):
    """Run generate_until against a fake server that echoes each prompt with its seed.

    Returns the generations, every payload sent (failed ones included) and the
    number of concurrent rounds, i.e. calls to `get_batched_requests`.
    """
    payloads = []

    def respond(payload):
        payloads.append(payload)
        if fail_first_request and len(payloads) == 1:
            raise ConnectionError("transient failure")
        prompts = payload["prompt"]
        prompts = [prompts] if isinstance(prompts, str) else prompts
        return {
            "choices": [
                {"index": i, "text": f"{p}|{payload['seed']}"}
                for i, p in enumerate(prompts)
            ]
        }

    def sync_post(*args, json, **kwargs):
        response = MagicMock(ok=True)
        response.json.return_value = respond(json)
        return response

    def async_post(*args, json, **kwargs):
        response = AsyncMock(ok=True, status=200)
        response.json = AsyncMock(return_value=respond(json))
        response.raise_for_status = lambda: None
        return DummyAsyncContextManager(response)

    with (
        patch("requests.post", side_effect=sync_post),
        patch("lm_eval.models.api_models.TCPConnector", autospec=True),
        patch("lm_eval.models.api_models.ClientSession", autospec=True) as session,
        patch("lm_eval.models.api_models.wait_exponential", return_value=wait_none()),
        patch.object(
            model, "get_batched_requests", wraps=model.get_batched_requests
        ) as get_batched_requests,
    ):
        session.return_value.__aenter__.return_value.post = async_post
        results = model.generate_until(instances)
    return results, payloads, get_batched_requests.call_count


@pytest.mark.parametrize("num_concurrent", [1, 3])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_generate_until_repeats_get_distinct_seeds(num_concurrent, batch_size):
    model = _completions_model(num_concurrent, batch_size)
    sampled = {"until": [], "do_sample": True, "temperature": 0.8}
    greedy = {"until": [], "do_sample": False, "temperature": 0}
    instances = (
        _repeated("a", sampled, 3)
        + _repeated("bb", sampled, 3)
        + _repeated("ccc", greedy, 2)
    )

    results, payloads, rounds = _generate_with_fake_server(model, instances)

    assert results == [
        "a|1234", "a|1235", "a|1236",
        "bb|1234", "bb|1235", "bb|1236",
        "ccc|1234", "ccc|1234",
    ]  # fmt: skip
    sizes = [1 if isinstance(p["prompt"], str) else len(p["prompt"]) for p in payloads]
    # each payload carries one seed, and requests that share a seed fill it
    assert sizes == [batch_size] * (len(instances) // batch_size)
    if num_concurrent > 1:
        # sampled repeats run in one concurrent round, not one round per seed
        assert rounds == 2  # one for the sampled requests, one for the greedy ones


def test_generate_until_repeats_offset_task_seed():
    gen_kwargs = {"until": [], "temperature": 0.8, "seed": 7}

    results, _, _ = _generate_with_fake_server(
        _completions_model(), _repeated("a", gen_kwargs, 2)
    )

    assert results == ["a|7", "a|8"]
    assert gen_kwargs == {"until": [], "temperature": 0.8, "seed": 7}


def test_generate_until_same_prompt_in_distinct_requests_keeps_model_seed():
    gen_kwargs = {"until": [], "temperature": 0.8}
    instances = _repeated("a", gen_kwargs, 1) + _repeated("a", gen_kwargs, 1)

    results, _, _ = _generate_with_fake_server(_completions_model(), instances)

    # distinct requests with the same prompt are not repeats
    assert results == ["a|1234", "a|1234"]


@pytest.mark.parametrize("num_concurrent", [1, 3])
def test_generate_until_retry_resends_the_same_seed(num_concurrent):
    gen_kwargs = {"until": [], "temperature": 0.8}

    results, payloads, _ = _generate_with_fake_server(
        _completions_model(num_concurrent),
        _repeated("a", gen_kwargs, 2),
        fail_first_request=True,
    )

    assert results == ["a|1234", "a|1235"]
    assert len(payloads) == 3
    assert payloads.count(payloads[0]) == 2  # the failed request is resent unchanged


def test_request_seed_is_sent_as_payload_seed_only():
    # SGLang spreads gen_kwargs into sampling_params, which has no `seed` field
    model = SGLANGGENERATEAPI(base_url="http://test-url.com", tokenizer_backend=None)
    gen_kwargs = {"until": [], "temperature": 0.8, "seed": 1235}

    with patch("requests.post") as mock_post:
        model.model_call(["a"], generate=True, gen_kwargs=gen_kwargs)

    assert "seed" not in mock_post.call_args.kwargs["json"]["sampling_params"]
    assert gen_kwargs["seed"] == 1235
